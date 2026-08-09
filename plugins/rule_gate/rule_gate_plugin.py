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
  R5 monitor-state  : a monitoring/limit-check state field (e.g. LC.CurrentLCState)
                      leaves its protective session baseline — the fault-management
                      / safe-mode disable step that turns off LC/HS/limit checking.
                      Shared, telemetry-indistinguishable footprint across the family
                      (LC.CurrentLCState 1→3): EX-0011 (exploit safe-mode), DE-0005
                      (subvert safe-mode), DE-0001 (disable fault management).
                      Validated 2026-07-16 (EX-0011/DE-0005) + 2026-07-29 (DE-0001,
                      AINOS3-73); blind to both the dynamics-IF and the other rules.
  R6 sb-command     : CFE_SB.CommandCounter reaches a new high — a CFE_SB command
                      (ENABLE/DISABLE_ROUTE, subscription report). Static in nominal
                      ops, so any command is the direct EX-0012.02 routing-table
                      signal — a lower-latency catch than the staleness gate's freeze
                      detection. A single-step signal held for a short dwell so the
                      leaky integrator latches one bounded incident per command.
  R7 evs-command    : CFE_EVS_HK.CommandCounter reaches a new high — a CFE_EVS command
                      (DISABLE_EVENT_TYPE = DE-0002.03 telemetry inhibit). Same
                      mechanism as R6 (static-in-nominal counter → new-high + dwell);
                      the low-latency command catch for the same freeze the staleness
                      gate detects via CFE_EVS_HK.MessageSendCounter going quiet.
  R8 es-command     : CFE_ES.CommandCounter reaches a new high — a CFE_ES (Executive
                      Services) command modifying a C&DH on-board value (EX-0012.10:
                      SET_MAX_PR_COUNT defeats the auto power-on-reset safeguard,
                      SET_PERF_FILTER_MASK, memory writes). Same static-in-nominal
                      new-high + dwell mechanism as R6/R7. The IF is CDH-blind and the
                      consistency/staleness gates miss it (the value goes UP, the
                      stream never freezes), so this rule is the only catch.
  R9 tbl-command    : CFE_TBL.CommandCounter reaches a new high — a CFE_TBL command
                      (LOAD/ACTIVATE = table-backdoor persistence, PER-0001). Static in
                      nominal (nothing loads tables in steady state). Same mechanism as
                      R6/R7/R8; the running-max survives the attacker's evidence-hiding
                      CFE_TBL_RESET (the new high is latched before the reset zeroes it).
  R10 bus-sweep     : META-rule — >= BusSweepMinRules (default 3) DISTINCT command rules
                      (R6-R9) fire in the same window. A single technique trips one
                      command counter; a flat-bus SWEEP (LM-0002) trips many at once.
                      Priced above the individual command rules so the incident labels
                      LM-0002 instead of collapsing to whichever single rule (usually
                      R2 evs-flood) outlasts the others.
  R11 fm-command    : FM.CommandCounter reaches a new high — a File Manager command
                      (COPY/MOVE/DELETE/DELETE_ALL). Static in nominal ops (nothing
                      routinely commands the filesystem in steady state; validated
                      2026-07-29 live: static at 0/1 while DS.FileWriteCounter climbs
                      continuously — DS is NOT usable, FM is). A burst of FM file ops is
                      the on-board footprint of a wiper (EX-0010.02 mass DELETE_ALL) or
                      ransomware (EX-0010.01 COPY->.enc + DELETE churn); the two are
                      indistinguishable at the HK level, so R11 catches the class and the
                      command mix disambiguates. Same static-in-nominal new-high + dwell
                      mechanism as R6-R9. The dynamics-IF is filesystem-blind and the
                      consistency/staleness gates miss it (the counter goes UP, no stream
                      freezes, no backward counter), so this rule is the only catch.
  R12 to-command    : TO.usCmdCnt reaches a new high — a command to the full Telemetry
                      Output app (TO_ENABLE_OUTPUT redirects the downlink to an attacker
                      = EXF-0003.02 downlink exfiltration). Static in nominal (the GSW
                      enables the downlink once at connect, then leaves it). Same
                      static-in-nominal new-high + dwell mechanism as R6-R11.
  R13 to-route      : TO.usEnabledRoutes / TO.usConfigRoutes (the downlink route masks)
                      leave their nominal baseline — the downlink routing/destination was
                      reconfigured (EXF-0003.02 exfil / IMP-0006 theft). A specific
                      corroborator of R12: fires on a route enable/disable/add; a
                      same-destination redirect that leaves the mask unchanged is still
                      caught by R12's command-counter increment. Baseline-deviation +
                      leaky-integrator flicker tolerance, same as R5 monitor-state.

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
    "R6:sb-command": "CFE_SB command — routing/subscription modification (EX-0012.02)",
    "R7:evs-command": "CFE_EVS command — event-type suppression / inhibit (DE-0002.03)",
    "R8:es-command": "CFE_ES command — C&DH on-board value modification (EX-0012.10)",
    "R9:tbl-command": "CFE_TBL command — table load/activate persistence (PER-0001)",
    "R10:bus-sweep": "Bus sweep — many MIDs commanded in one window (LM-0002 lack of bus segregation)",
    "R11:fm-command": "File Manager command — file-operation burst (EX-0010.01 ransomware / EX-0010.02 wiper)",
    "R12:to-command": "Telemetry Output command — downlink reconfigure (EXF-0003.02 downlink exfiltration)",
    "R13:to-route": "Downlink route mask changed — downlink reconfigured (EXF-0003.02 exfil / IMP-0006 theft)",
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


def _monstate_label(rule_id: str) -> str:
    """Human label for an R5 monitor-state alert (rule-id `R5:{app}-monstate`).

    The LC-disable footprint (CurrentLCState → DISABLED) is shared and
    telemetry-indistinguishable across the fault-management/safe-mode family:
    EX-0011 (exploit safe-mode), DE-0005 (subvert safe-mode), and DE-0001
    (disable fault management). All three drive LC.CurrentLCState 1→3; they
    differ only by accompanying steps (EX-0011 adds thruster physics the IF
    catches; DE-0005 forces an ADCS mode; DE-0001 is LC/HS-only). Validated
    live 2026-07-29 (AINOS3-73): DE-0001's LC-disable latches this same R5."""
    app = rule_id.split(":", 1)[1].rsplit("-", 1)[0]
    return (f"{app} monitoring/limit-check state left its protective baseline "
            f"(fault-management disable — DE-0001 / EX-0011 safe-mode "
            f"induction / DE-0005)")


def _incident_label(active_rules):
    """(cluster, sub_technique) for the primary active rule — the incident label.

    Priority device-disable > bus-sweep > monitor-state/route-state >
    sb/evs/es/tbl/fm/to-command > evs-flood > cmd-error > sb-error, so a DE-0010 flood
    (R2+R3) labels DE-0010, a downlink route change (R12+R13) labels EXF-0003.02/to-route, a
    sensor disable as its technique, an LC/HS monitoring-disable as EX-0011, and a
    multi-command sweep as LM-0002 (rather than collapsing to its loudest single rule).
    """
    def _prio(r):
        return {"R1": 0, "R10": 1, "R5": 2, "R13": 2, "R6": 3, "R7": 3, "R8": 3,
                "R9": 3, "R11": 3, "R12": 3, "R2": 4, "R4": 5, "R3": 6}.get(
                    r.split(":", 1)[0], 7)
    if not active_rules:
        return "", ""
    r = min(active_rules, key=_prio)
    if r.startswith("R1:"):
        comp = r.split(":", 1)[1].rsplit("-", 1)[0]
        cluster = ("EX-0002" if comp.startswith("NOVATEL")
                   else "EX-0014.03" if comp in ("IMU", "MAG", "CSS", "FSS", "ST")
                   else "DE-0002.03")
        return cluster, f"{comp}-disabled"
    if r == "R10:bus-sweep":
        return "LM-0002", "bus-sweep"
    if r.startswith("R5:"):
        app = r.split(":", 1)[1].rsplit("-", 1)[0]
        return "EX-0011", f"{app}-monitoring-disabled"
    if r == "R6:sb-command":
        return "EX-0012.02", "sb-command"
    if r == "R7:evs-command":
        return "DE-0002.03", "evs-command"
    if r == "R8:es-command":
        return "EX-0012.10", "es-command"
    if r == "R9:tbl-command":
        return "PER-0001", "tbl-command"
    if r == "R11:fm-command":
        return "EX-0010", "fm-command"
    if r == "R12:to-command":
        return "EXF-0003.02", "to-command"
    if r == "R13:to-route":
        return "EXF-0003.02", "to-route"
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
        # R5 monitor-state: comma-separated telemetry fields whose deviation from
        # their protective session baseline is an alert (monitoring/limit-check
        # turned off). LC.CurrentLCState: 1=ACTIVE(protective) 2=PASSIVE 3=DISABLED.
        "MonitorStateFields": "LC.CurrentLCState",
        # R13 route-state: comma-separated telemetry fields whose deviation from their
        # nominal session baseline means the downlink was reconfigured (EXF-0003.02).
        # The full TO app's route masks are static in nominal; a change = exfil signal.
        "RouteStateFields": "TO.usEnabledRoutes,TO.usConfigRoutes",
        # R6/R7 command rules: CFE_SB.CommandCounter and CFE_EVS_HK.CommandCounter are
        # STATIC in nominal ops (nothing routinely commands the Software Bus or Event
        # Services), so any increment = an attacker command — R6 CFE_SB
        # (ENABLE/DISABLE_ROUTE = EX-0012.02 routing), R7 CFE_EVS (DISABLE_EVENT_TYPE =
        # DE-0002.03 inhibit). A command is a single new-high step, so we hold the fire
        # for CmdDwell frames to let the leaky integrator latch one bounded incident.
        "CmdDwell": "8",
        # R10 bus-sweep meta-rule: LM-0002 sweeps every reachable MID, tripping several
        # static-in-nominal command counters (R6/R7/R8/R9) in the same short window. No
        # single technique does that, so when this many DISTINCT command rules are firing
        # together, label it a bus sweep (LM-0002) rather than letting it collapse to
        # whichever single rule (usually R2 evs-flood) outlasts the others.
        "BusSweepMinRules": "3",
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
        # R6/R7/R8/R9/R11/R12 command rules: a `*.CommandCounter` (or TO.usCmdCnt) that
        # is STATIC in nominal ops (nothing routinely commands the Software Bus / Event
        # Services / Executive Services / Table Services / File Manager / Telemetry
        # Output), so any new high is an attacker command. col index → rule-id (present
        # fields only). FM.CommandCounter is the wiper/ransomware (EX-0010) file-op-burst
        # signal (R11); TO.usCmdCnt is the downlink-exfil (EXF-0003.02) signal (R12).
        self._cmd_dwell_frames = _int("CmdDwell")
        self._bus_sweep_min = _int("BusSweepMinRules")
        self._cmd_rule = {}
        for field, rid in (("CFE_SB.CommandCounter", "R6:sb-command"),
                           ("CFE_EVS_HK.CommandCounter", "R7:evs-command"),
                           ("CFE_ES.CommandCounter", "R8:es-command"),
                           ("CFE_TBL.CommandCounter", "R9:tbl-command"),
                           ("FM.CommandCounter", "R11:fm-command"),
                           ("TO.usCmdCnt", "R12:to-command")):
            if field in idx:
                self._cmd_rule[idx[field]] = rid
        # R5 monitor-state fields present in this schema, name → column index.
        monstate_cfg = cfg.get("monitorstatefields", self.DEFAULTS["MonitorStateFields"])
        self._monstate_idx = {f.strip(): idx[f.strip()]
                              for f in monstate_cfg.split(",")
                              if f.strip() and f.strip() in idx}
        # R13 route-state (EXF-0003.02 corroborator): the full TO app's downlink
        # route masks (TO.usEnabledRoutes / TO.usConfigRoutes) are STATIC in nominal;
        # a change means the downlink routing/destination was reconfigured — the
        # specific downlink-exfil signal that complements R12's command counter.
        # (Fires on an enable/disable/add-route; a same-destination redirect that
        # leaves the mask unchanged is still caught by R12's usCmdCnt increment.)
        route_cfg = cfg.get("routestatefields", self.DEFAULTS["RouteStateFields"])
        self._routestate_idx = {f.strip(): idx[f.strip()]
                                for f in route_cfg.split(",")
                                if f.strip() and f.strip() in idx}

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
        for f in self._monstate_idx:
            self._rule_ids.add(f"R5:{f.split('.')[0]}-monstate")
        for rid in self._cmd_rule.values():
            self._rule_ids.add(rid)
        if self._routestate_idx:
            self._rule_ids.add("R13:to-route")
        # R10 bus-sweep meta-rule can only fire if enough command counters exist to
        # cross the threshold; register it only then.
        if len(self._cmd_rule) >= self._bus_sweep_min:
            self._rule_ids.add("R10:bus-sweep")

        self._activity = {r: 0.0 for r in self._rule_ids}
        self._rule_active = {r: False for r in self._rule_ids}
        self._enable_baseline = {h: 0.0 for h in self._enable_idx}
        # R5 baseline = the protective state established during warmup (first seen,
        # e.g. LC ACTIVE=1). None until a real value arrives; a later deviation fires.
        self._monstate_baseline = {f: None for f in self._monstate_idx}
        # R13 route-state baseline = the downlink route mask at warmup (static in
        # nominal). None until a real value arrives; a later deviation fires.
        self._routestate_baseline = {f: None for f in self._routestate_idx}
        # R6/R7: per command-counter running max (a new high = a new command) + a
        # dwell countdown so a single-step command latches the leaky integrator.
        self._cmd_max = {i: None for i in self._cmd_rule}
        self._cmd_dwell = {i: 0 for i in self._cmd_rule}
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
              f"{len(self._cmd_rule)} static-cmd counters, "
              f"{len(self._monstate_idx)} monitor-state fields "
              f"({', '.join(self._monstate_idx) or 'none'}), "
              f"{len(self._routestate_idx)} route-state fields "
              f"({', '.join(self._routestate_idx) or 'none'}), "
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
        # R5 monitor-state: a watched monitoring/limit-check state field left its
        # protective baseline (captured on first receipt, e.g. LC.CurrentLCState
        # ACTIVE=1). A discrete flip like 1→3 (DISABLED) fires; a return to baseline
        # clears via the leaky integrator, same as R1.
        for f, i in self._monstate_idx.items():
            v = val(i)
            if v is None:
                continue
            if self._monstate_baseline[f] is None:
                self._monstate_baseline[f] = v   # establish the protective baseline
            elif v != self._monstate_baseline[f]:
                fired.add(f"R5:{f.split('.')[0]}-monstate")
        # R13 route-state: the full TO app's downlink route mask left its nominal
        # baseline — the downlink was reconfigured (route enabled/disabled/added). A
        # specific downlink-exfil signal (EXF-0003.02) that corroborates R12. Same
        # baseline-deviation + leaky-integrator flicker tolerance as R5.
        for f, i in self._routestate_idx.items():
            v = val(i)
            if v is None:
                continue
            if self._routestate_baseline[f] is None:
                self._routestate_baseline[f] = v
            elif v != self._routestate_baseline[f]:
                fired.add("R13:to-route")
        # R6/R7 command rules: a NEW HIGH of a static `*.CommandCounter` = an attacker
        # command was processed (R6 CFE_SB → routing/subscription modification; R7
        # CFE_EVS → event-type suppression / DE-0002.03 inhibit). The running max
        # ignores the double-buffer flicker back to stale values; the dwell holds the
        # fire long enough for the leaky integrator to latch one bounded incident.
        for i, rid in self._cmd_rule.items():
            v = val(i)
            if v is not None:
                if self._cmd_max[i] is None:
                    self._cmd_max[i] = v          # establish baseline (warmup)
                elif v > self._cmd_max[i]:
                    self._cmd_max[i] = v
                    self._cmd_dwell[i] = self._cmd_dwell_frames
            if self._cmd_dwell[i] > 0:
                fired.add(rid)
                self._cmd_dwell[i] -= 1

        # R10 bus-sweep (LM-0002): several distinct static-in-nominal command counters
        # tripped in the same window. The individual command dwells overlap during a
        # sweep, so counting the distinct command rules firing THIS frame catches it.
        if "R10:bus-sweep" in self._rule_ids:
            n_cmd = sum(1 for rid in self._cmd_rule.values() if rid in fired)
            if n_cmd >= self._bus_sweep_min:
                fired.add("R10:bus-sweep")

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
            if r.startswith("R1:"):
                label = _device_label(r)
            elif r.startswith("R5:"):
                label = _monstate_label(r)
            else:
                label = _TECH_LABEL.get(r) or r.split(":", 1)[1]
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
