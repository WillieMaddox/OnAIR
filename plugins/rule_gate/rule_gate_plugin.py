# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""Rule-gate learner plugin — runs PARALLEL to the isolation_forest IF.

The Section-A validation (2026-07-16) established that the v5 IF is a *dynamics*
detector: it flags anomalies in the GNC/attitude physics it is trained on, but is
structurally blind to attacks whose footprint is a discrete flag flip or a
counter/rate spike that doesn't disturb the physics (DE-0010 EVS flood, EX-0002
GPS disable, EX-0014.03 IMU disable — all scored is_anomaly=0). Because the
classifier is IF-gated, those go undetected end-to-end despite loud, subscribed
signals.

This plugin is the complementary gate. It reads raw telemetry directly (never
the IF's high_level_data) and fires on:
  R1 device-disable : any `*.DeviceEnabled` drops below its session baseline.
  R2 evs-flood      : CFE_EVS_HK.MessageSendCounter per-frame delta > threshold.
  R3 sb-errors      : CFE_SB.MsgSendErrorCounter per-frame delta > 0.
  R4 cmd-errors     : any `*.CommandError{Count,Counter}` per-frame delta > thresh.

The rule that fires IS the label (R1:NOVATEL → GPS disable, etc.), so no XGBoost
classifier is needed for this class. The two gates are complementary: the IF owns
dynamics attacks (subsystem-value corruption, ROBUST tier); this owns state-change
attacks. An operator OR's the two alerts.

HYSTERESIS — leaky integrator, NOT the IF's consecutive-frame counter. The OnAIR
double-buffer makes a discrete signal flicker: a disabled DeviceEnabled shows 0 in
the buffer that got the fresh HK and stale 1 in the other, and a rate-delta is
nonzero only on the frame new data lands — so R2 fires every *other* frame.
Consecutive-N hysteresis would be defeated by that. Instead each rule keeps a
leaky activity counter (+FireIncrement on a fire, −Decay on a quiet frame, clamped
[0, ActivityCap]); the operational alert latches on at AlertLevel and off at
ClearLevel. This tolerates every-other-frame flicker and still clears cleanly.

Emits a sibling side-file `rule_gate_out_<ts>_pid<N>.csv` next to the IF's
`iforest_out_*.csv`, joined by pid + cumulative row index.
"""
from __future__ import annotations

import configparser
import csv
import datetime
import os
import re
import sys as _sys

from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin

# Reuse the incident aggregator (NOS3-201) that ships in the sibling
# xgb_classifier plugin package, so rule-gate alerts fold into the SAME Incident
# format the IF→classifier path uses (operators OR the two incident streams).
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xgb_classifier"))
from incident import IncidentAggregator  # noqa: E402


# ADCS mode enum → name (matches run_attack.py / the IF routing).
_MODE_NAME = {0: "PASSIVE", 1: "BDOT", 2: "SUNSAFE", 3: "INERTIAL"}


# rule-id prefix → operator-readable technique label
_TECH_LABEL = {
    "R2:evs": "EVS event-log flood (DE-0010 overflow-audit-log)",
    "R3:sb": "Software Bus send errors",
}


def _to_num(value):
    """Parse a telemetry cell to float; None for the `[0]` init sentinel / text."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _device_label(rule_id: str) -> str:
    comp = rule_id.split(":", 1)[1].rsplit("-", 1)[0]
    if comp.startswith("NOVATEL"):
        return f"{comp} GPS receiver disabled (EX-0002 PNT geofencing)"
    if comp in ("IMU", "MAG", "CSS", "FSS", "ST"):
        return f"{comp} attitude sensor disabled (EX-0014.03 sensor spoof)"
    return f"{comp} subsystem disabled (DE-0002.03 inhibit)"


def _incident_label(active_rules):
    """(cluster, sub_technique) for the primary active rule — the incident label.

    Priority device-disable > evs-flood > cmd-error > sb-error, so a DE-0010 flood
    (which fires R2+R3) labels as DE-0010, and a sensor disable as its technique.
    """
    def _prio(r):
        return {"R1": 0, "R2": 1, "R4": 2, "R3": 3}.get(r.split(":", 1)[0], 4)
    if not active_rules:
        return "", ""
    r = min(active_rules, key=_prio)
    if r.startswith("R1:"):
        comp = r.split(":", 1)[1].rsplit("-", 1)[0]
        cluster = ("EX-0002" if comp.startswith("NOVATEL")
                   else "EX-0014.03" if comp in ("IMU", "MAG", "CSS", "FSS", "ST")
                   else "DE-0002.03")
        return cluster, f"{comp}-disabled"
    if r == "R2:evs":
        return "DE-0010", "evs-flood"
    if r == "R3:sb":
        return "DE-0010", "sb-send-errors"
    return "cmd-errors", r.split(":", 1)[1]


class Plugin(AIPlugin):
    """Rule/threshold gate for state-change attacks the dynamics-IF misses."""

    DEFAULTS = {
        "WriteSideFile": "true",
        "SideFileOutputDir": "../../../../data/onair/csv",  # mirrors IF/csv_output
        "SideFileFlushEvery": "10",
        # Startup-transient suppression: the first frames after connect have
        # not-yet-arrived MIDs (placeholder + huge first deltas). During warmup
        # we build the DeviceEnabled baseline and prime deltas, but never alert.
        "WarmupFrames": "30",
        # Leaky-integrator hysteresis (see module docstring).
        "FireIncrement": "2",
        "Decay": "1",
        "ActivityCap": "8",
        "AlertLevel": "4",
        "ClearLevel": "1",
        # Rule thresholds.
        "EvsRateThreshold": "15",   # events/frame above nominal background
        "SbErrThreshold": "0",      # any SB send-error increment
        "CmdErrThreshold": "3",     # command errors/frame
        "HeartbeatEvery": "1000",
        # Incident aggregation (NOS3-201). The rule-gate's leaky integrator
        # already smoothed flicker, so the incident layer's own hysteresis is
        # small — it just folds a sustained alert into one labeled incident.
        "WriteIncidentFile": "true",
        "IncidentAlertHysteresis": "1",
        "IncidentClearHysteresis": "3",
        "IncidentMinAnomalyFrames": "3",
    }

    def __init__(self, name, headers):
        super().__init__(name, headers)
        cfg = self._load_config()

        def _int(k):
            return int(cfg.get(k.lower(), self.DEFAULTS[k]))

        self._warmup_frames = _int("WarmupFrames")
        self._fire_inc = _int("FireIncrement")
        self._decay = _int("Decay")
        self._cap = _int("ActivityCap")
        self._alert_level = _int("AlertLevel")
        self._clear_level = _int("ClearLevel")
        self._evs_thresh = float(cfg.get("evsratethreshold", self.DEFAULTS["EvsRateThreshold"]))
        self._sb_thresh = float(cfg.get("sberrthreshold", self.DEFAULTS["SbErrThreshold"]))
        self._cmderr_thresh = float(cfg.get("cmderrthreshold", self.DEFAULTS["CmdErrThreshold"]))
        self._heartbeat_every = _int("HeartbeatEvery")

        # Resolve the telemetry columns each rule watches, by name → index into
        # low_level_data (headers and low_level_data are index-aligned).
        idx = {h: i for i, h in enumerate(self.headers)}
        self._enable_idx = {h: i for h, i in idx.items() if h.endswith(".DeviceEnabled")}
        self._cmderr_idx = {h: i for h, i in idx.items()
                            if re.search(r"CommandError(Count|Counter)$", h)}
        self._evs_idx = idx.get("CFE_EVS_HK.MessageSendCounter")
        self._sb_idx = idx.get("CFE_SB.MsgSendErrorCounter")
        self._mode_idx = idx.get("ADCS_GNC.Mode")

        # The full set of rule-ids that can ever fire (so the leaky counter
        # decays even on frames a rule is quiet).
        self._rule_ids = set()
        for h in self._enable_idx:
            self._rule_ids.add(f"R1:{h.split('.')[0]}-disabled")
        if self._evs_idx is not None:
            self._rule_ids.add("R2:evs")
        if self._sb_idx is not None:
            self._rule_ids.add("R3:sb")
        for h in self._cmderr_idx:
            self._rule_ids.add(f"R4:{h.split('.')[0]}-cmderr")

        self._activity = {r: 0.0 for r in self._rule_ids}
        self._rule_active = {r: False for r in self._rule_ids}
        self._enable_baseline = {h: 0.0 for h in self._enable_idx}
        self._prev_evs = None
        self._prev_sb = None
        self._prev_cmderr = {h: None for h in self._cmderr_idx}
        self._frame_count = 0
        self._latest_active = []

        # Incident aggregation (NOS3-201): fold rule-gate alerts into the same
        # labeled Incident format the IF→classifier path emits.
        self._incident_agg = IncidentAggregator(
            alert_hysteresis=_int("IncidentAlertHysteresis"),
            clear_hysteresis=_int("IncidentClearHysteresis"),
            min_anomaly_frames=_int("IncidentMinAnomalyFrames"))
        self._incident_file_path = None
        self._incident_header_written = False
        self._n_incidents = 0

        # Side-file + incident-file setup (mirrors the IF/classifier plugins).
        self._side_file_path = None
        self._side_file_buffer = []
        self._side_file_header_written = False
        self._side_file_flush_every = _int("SideFileFlushEvery")
        out_dir = cfg.get("sidefileoutputdir", self.DEFAULTS["SideFileOutputDir"])
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
        if cfg.get("writesidefile", self.DEFAULTS["WriteSideFile"]).strip().lower() == "true":
            os.makedirs(out_dir, exist_ok=True)
            self._side_file_path = os.path.join(out_dir, f"rule_gate_out_{ts}_pid{os.getpid()}.csv")
        if cfg.get("writeincidentfile", self.DEFAULTS["WriteIncidentFile"]).strip().lower() == "true":
            os.makedirs(out_dir, exist_ok=True)
            self._incident_file_path = os.path.join(out_dir, f"rule_gate_incident_{ts}_pid{os.getpid()}.csv")

        print(f"[rule_gate] watching {len(self._enable_idx)} DeviceEnabled flags, "
              f"{len(self._cmderr_idx)} cmd-err counters, "
              f"EVS={'y' if self._evs_idx is not None else 'n'} "
              f"SB={'y' if self._sb_idx is not None else 'n'}; "
              f"warmup={self._warmup_frames}, leaky "
              f"(+{self._fire_inc}/-{self._decay}, alert>={self._alert_level}, clear<={self._clear_level})")
        if self._side_file_path is not None:
            print(f"[rule_gate] side-file → {self._side_file_path}")
        if self._incident_file_path is not None:
            print(f"[rule_gate] incident-file → {self._incident_file_path} "
                  f"(alert/clear hyst {self._incident_agg.alert_hyst}/{self._incident_agg.clear_hyst})")

    @staticmethod
    def _load_config():
        ini_path = os.environ.get("ONAIR_INI_FILE") or "cf/onair/nos3_security.ini"
        if not os.path.exists(ini_path):
            return {}
        parser = configparser.ConfigParser()
        parser.read(ini_path)
        return dict(parser.items("RULE_GATE")) if parser.has_section("RULE_GATE") else {}

    def update(self, low_level_data=[], high_level_data={}):
        self._frame_count += 1

        def val(i):
            return _to_num(low_level_data[i]) if i is not None and i < len(low_level_data) else None

        in_warmup = self._frame_count <= self._warmup_frames

        # During warmup, learn the DeviceEnabled baseline (max seen) and prime
        # the counters — but do not raise/clear alerts.
        for h, i in self._enable_idx.items():
            v = val(i)
            if v is not None and in_warmup:
                self._enable_baseline[h] = max(self._enable_baseline[h], v)

        # ── Per-frame raw rule fires ────────────────────────────────────────
        fired = set()
        # R1 device-disable
        for h, i in self._enable_idx.items():
            v = val(i)
            if v is not None and self._enable_baseline[h] > 0 and v < self._enable_baseline[h]:
                fired.add(f"R1:{h.split('.')[0]}-disabled")
        # R2 evs-flood
        v = val(self._evs_idx)
        if v is not None and self._prev_evs is not None and (v - self._prev_evs) > self._evs_thresh:
            fired.add("R2:evs")
        if v is not None:
            self._prev_evs = v
        # R3 sb-errors
        v = val(self._sb_idx)
        if v is not None and self._prev_sb is not None and (v - self._prev_sb) > self._sb_thresh:
            fired.add("R3:sb")
        if v is not None:
            self._prev_sb = v
        # R4 cmd-errors
        for h, i in self._cmderr_idx.items():
            v = val(i)
            if v is not None and self._prev_cmderr[h] is not None and (v - self._prev_cmderr[h]) > self._cmderr_thresh:
                fired.add(f"R4:{h.split('.')[0]}-cmderr")
            if v is not None:
                self._prev_cmderr[h] = v

        # ── Leaky-integrator hysteresis + edge-triggered alert/clear ────────
        edge = []
        for r in self._rule_ids:
            if r in fired:
                self._activity[r] = min(self._cap, self._activity[r] + self._fire_inc)
            else:
                self._activity[r] = max(0.0, self._activity[r] - self._decay)
            if not in_warmup:
                if not self._rule_active[r] and self._activity[r] >= self._alert_level:
                    self._rule_active[r] = True
                    edge.append(("ALERT", r))
                elif self._rule_active[r] and self._activity[r] <= self._clear_level:
                    self._rule_active[r] = False
                    edge.append(("CLEAR", r))

        self._latest_active = sorted(r for r, a in self._rule_active.items() if a)

        # ── Fold the alert into a labeled incident (NOS3-201) ───────────────
        # R3 (SB send-errors) is a noisy background artifact in NOS3 — a stack
        # with an unconnected downlink spams RADIO device-HK failures that climb
        # CFE_SB.MsgSendErrorCounter continuously (measured ~8/frame nominal). It
        # stays in the alert stream / side-file as a corroborator, but it must
        # NOT drive an incident on its own, or the incident never closes. The
        # incident is driven by the *specific* rules (R1 device-disable, R2
        # EVS-flood, R4 cmd-errors).
        if not in_warmup:
            incident_active = [r for r in self._latest_active if not r.startswith("R3:")]
            mode = _MODE_NAME.get(int(m), str(m)) if (m := val(self._mode_idx)) is not None else ""
            cluster, sub = _incident_label(incident_active)
            closed = self._incident_agg.update(
                self._frame_count - 1, bool(incident_active),
                mode=mode, cluster=cluster, sub_technique=sub,
                confidence=1.0 if incident_active else 0.0)
            if closed is not None:
                self._write_incident(closed)

        for kind, r in edge:
            label = _TECH_LABEL.get(r) or (_device_label(r) if r.startswith("R1:")
                                           else r.split(":", 1)[1])
            print(f"[rule_gate][{kind}] frame={self._frame_count} rule={r} — {label}")
        if not edge and self._heartbeat_every > 0 and self._frame_count % self._heartbeat_every == 0:
            tag = " [warmup]" if in_warmup else ""
            print(f"[rule_gate] frame={self._frame_count}{tag} "
                  f"active={self._latest_active or 'none'}")

        if self._side_file_path is not None:
            self._side_file_buffer.append([
                self._frame_count - 1,
                ";".join(self._latest_active),
                int(bool(self._latest_active)),
                int(bool(edge)),
            ])
            if len(self._side_file_buffer) >= self._side_file_flush_every:
                self._flush_side_file()

    def render_reasoning(self):
        return {
            "rule_gate_alert": bool(self._latest_active),
            "active_rules": self._latest_active,
            "n_incidents": self._n_incidents,
            "frame": self._frame_count,
        }

    def _write_incident(self, incident):
        """Append a closed rule-gate incident (same Incident row format as the
        IF→classifier incident file, so the two streams merge downstream)."""
        self._n_incidents += 1
        print(f"[rule_gate][INCIDENT] #{self._n_incidents} "
              f"frames {incident.frame_start}-{incident.frame_end} "
              f"({incident.n_frames}f) mode={incident.mode} "
              f"cluster={incident.cluster} sub={incident.sub_technique}")
        if self._incident_file_path is None:
            return
        write_header = not self._incident_header_written
        with open(self._incident_file_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(incident.header())
                self._incident_header_written = True
            w.writerow(incident.as_row())

    def _flush_side_file(self):
        if self._side_file_path is None or not self._side_file_buffer:
            return
        write_header = not self._side_file_header_written
        with open(self._side_file_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["frame_idx", "active_rules", "alert_active", "edge"])
                self._side_file_header_written = True
            w.writerows(self._side_file_buffer)
        self._side_file_buffer = []
