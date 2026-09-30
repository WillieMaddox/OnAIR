# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""Consistency-check learner — the THIRD detector gate, for bus-traffic spoofing.

The EX-0014.02 validation (2026-07-16) established a class of attack both existing
gates miss: an external attacker injects a spoofed telemetry packet onto the SB via
CI_LAB (:5012), which republishes ANY MID with no command/telemetry filter. The
spoof is a *per-sample, out-of-distribution* value:
  - the dynamics-IF is blind to it (a spoofed counter/flag doesn't perturb the GNC
    physics the IF watches), and
  - the rule-gate's leaky integrator can't LATCH it (the real publisher overwrites
    the spoof within 1-2 frames, so it flickers and never reaches AlertLevel).

This gate is per-SAMPLE. Its primitive is **monotonic-counter violation**: many cFS
telemetry fields are monotonically-increasing counters, and a spoofed packet injects
a value off that trajectory (the DEMONSTRATED spoof set a counter backwards to a
mid-range value).

THE HARD PART is OnAIR's double buffer. Reading it per-frame makes every fast field
alternate between two buffer values (e.g. LC.MonitoredMsgCount oscillates
45145↔45138) — a naive "prev - new > threshold" back-step check false-positives on
~83% of frames. Three defenses, tuned against a live nominal baseline:
  1. NAME filter — only fields named like counters (Count/Counter/Packets). Excludes
     oscillating PHYSICAL values (MagneticIntensity, rates) that a short warmup can
     mistake for monotonic.
  2. ROLLING-WINDOW-MIN — flag `new < min(recent window) - Margin`, not `new < prev`.
     Double-buffer flicker oscillates between two recent values, so `new` is never
     below the window minimum; a spoofed outlier is. This is the key FP killer.
  3. uint8 EXCLUSION — counters whose observed max ≤ 255 wrap (255→0) and are noisy;
     watch only wide (uint32) counters that don't wrap in a session. A drop to ~0
     (DropFloor) is also excluded as a reset/wrap.

Emits a sibling side-file `consistency_out_<ts>_pid<N>.csv` and, via the shared
IncidentAggregator (AINOS3-25), labeled EX-0014.02 incidents — single-sample
sensitive (IncidentMinAnomalyFrames=1) because spoofs are transient. Operators OR
this incident stream with the IF→classifier and rule-gate streams.
"""
from __future__ import annotations

import collections
import configparser
import csv
import datetime
import os
import re
import sys as _sys

from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin

# Reuse the incident aggregator (AINOS3-25) from the sibling xgb_classifier package
# so spoof detections fold into the SAME Incident format the other gates emit.
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xgb_classifier"))
from incident import IncidentAggregator  # noqa: E402


# ADCS mode enum → name (matches the IF routing / rule_gate).
_MODE_NAME = {0: "PASSIVE", 1: "BDOT", 2: "SUNSAFE", 3: "INERTIAL"}

# A field is counter-like if its name carries a count/packet token. This excludes
# physical measurements (MagneticIntensity, voltages, rates) that a 30-frame warmup
# can otherwise mistake for a monotonic series.
_COUNTER_NAME = re.compile(r"(Count|Counter|Packets)", re.IGNORECASE)


def _to_num(value):
    """Parse a telemetry cell to float; None for the `[0]` init sentinel / text."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Plugin(AIPlugin):
    """Per-sample consistency checker for bus-traffic spoofing (EX-0014.02)."""

    DEFAULTS = {
        "WriteSideFile": "true",
        "SideFileOutputDir": "../../../../data/csv",
        "SideFileFlushEvery": "10",
        "WarmupFrames": "30",
        # "auto" = discover wide monotonic counters during warmup, or a comma list.
        "MonotonicFields": "auto",
        # Rolling-window-min flicker rejection: flag new < min(window) - Margin.
        "WindowSize": "8",
        "Margin": "4",
        # Exclude reset/wrap to near-0, and uint8 counters (max <= 255) that wrap.
        "DropFloor": "8",
        "MinCounterMax": "256",
        "HeartbeatEvery": "1000",
        # Incident aggregation — single-sample sensitive: a spoof is transient, so
        # one violation frame is enough to open+close a (short) labeled incident.
        "WriteIncidentFile": "true",
        "IncidentAlertHysteresis": "1",
        "IncidentClearHysteresis": "1",
        "IncidentMinAnomalyFrames": "1",
    }

    def __init__(self, name, headers):
        super().__init__(name, headers)
        cfg = self._load_config()

        def _int(k):
            return int(cfg.get(k.lower(), self.DEFAULTS[k]))

        self._warmup_frames = _int("WarmupFrames")
        self._win_size = _int("WindowSize")
        self._margin = float(cfg.get("margin", self.DEFAULTS["Margin"]))
        self._drop_floor = float(cfg.get("dropfloor", self.DEFAULTS["DropFloor"]))
        self._min_max = float(cfg.get("mincountermax", self.DEFAULTS["MinCounterMax"]))
        self._heartbeat_every = _int("HeartbeatEvery")

        mf = cfg.get("monotonicfields", self.DEFAULTS["MonotonicFields"]).strip()
        idx = {h: i for i, h in enumerate(self.headers)}
        self._auto = (mf.lower() == "auto")
        if self._auto:
            # Candidate columns = counter-named fields; narrowed to wide monotonic
            # ones at the end of warmup.
            self._candidates = {i for i, h in enumerate(self.headers)
                                if _COUNTER_NAME.search(h)}
            self._watched = set()
        else:
            self._candidates = set()
            self._watched = {idx[f.strip()] for f in mf.split(",")
                             if f.strip() and f.strip() in idx}

        # Warmup discovery state (per candidate index).
        self._saw_inc = {}
        self._max_seen = {}
        self._bad = {}
        # Rolling window per index. In auto mode every candidate's window is PRIMED
        # during warmup so the first post-warmup sample already has recent context
        # (an empty window would false-flag the first flicker-low).
        self._win = {i: collections.deque(maxlen=self._win_size)
                     for i in (self._candidates or self._watched)}
        self._frame_count = 0
        self._latest_violations = []

        # Incident aggregation (AINOS3-25).
        self._incident_agg = IncidentAggregator(
            alert_hysteresis=_int("IncidentAlertHysteresis"),
            clear_hysteresis=_int("IncidentClearHysteresis"),
            min_anomaly_frames=_int("IncidentMinAnomalyFrames"))
        self._incident_file_path = None
        self._incident_header_written = False
        self._n_incidents = 0
        self._mode_idx = idx.get("ADCS_GNC.Mode")

        # Side-file + incident-file setup (mirrors the IF / rule_gate plugins).
        self._side_file_path = None
        self._side_file_buffer = []
        self._side_file_header_written = False
        self._side_file_flush_every = _int("SideFileFlushEvery")
        out_dir = cfg.get("sidefileoutputdir", self.DEFAULTS["SideFileOutputDir"])
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
        if cfg.get("writesidefile", self.DEFAULTS["WriteSideFile"]).strip().lower() == "true":
            os.makedirs(out_dir, exist_ok=True)
            self._side_file_path = os.path.join(out_dir, f"consistency_out_{ts}_pid{os.getpid()}.csv")
        if cfg.get("writeincidentfile", self.DEFAULTS["WriteIncidentFile"]).strip().lower() == "true":
            os.makedirs(out_dir, exist_ok=True)
            self._incident_file_path = os.path.join(out_dir, f"consistency_incident_{ts}_pid{os.getpid()}.csv")

        watched_note = (f"{len(self._candidates)} counter-named candidates"
                        if self._auto else f"{len(self._watched)} forced")
        print(f"[consistency_check] monotonic-counter check ({watched_note}); "
              f"flag new < window({self._win_size})-min - {self._margin:.0f}, "
              f"mid (>{self._drop_floor:.0f}), wide (max>{self._min_max:.0f}); "
              f"warmup={self._warmup_frames}")
        if self._side_file_path is not None:
            print(f"[consistency_check] side-file → {self._side_file_path}")
        if self._incident_file_path is not None:
            print(f"[consistency_check] incident-file → {self._incident_file_path}")

    @staticmethod
    def _load_config():
        ini_path = os.environ.get("ONAIR_INI_FILE") or "cf/onair/nos3_security.ini"
        if not os.path.exists(ini_path):
            return {}
        parser = configparser.ConfigParser()
        parser.read(ini_path)
        return dict(parser.items("CONSISTENCY_CHECK")) if parser.has_section("CONSISTENCY_CHECK") else {}

    def _violation(self, i, v):
        """True if v is a spoof-like outlier below the recent window for column i:
        below window-min by > Margin (clears double-buffer flicker) and mid-range
        (not a reset/wrap toward 0)."""
        win = self._win.get(i)
        if not win or v <= self._drop_floor:
            return False
        return v < (min(win) - self._margin)

    def update(self, low_level_data=[], high_level_data={}):
        self._frame_count += 1
        n = len(low_level_data)
        in_warmup = self._frame_count <= self._warmup_frames

        if in_warmup:
            if self._auto:
                for i in self._candidates:
                    v = _to_num(low_level_data[i]) if i < n else None
                    if v is None:
                        continue
                    prev_max = self._max_seen.get(i)
                    if prev_max is not None and v > prev_max:
                        self._saw_inc[i] = True
                    # a mid-range back-step during warmup (below the primed window)
                    # disqualifies the field as a clean counter
                    if self._violation(i, v):
                        self._bad[i] = True
                    self._max_seen[i] = max(v, self._max_seen.get(i, v))
                    self._win[i].append(v)   # prime the rolling window
                if self._frame_count == self._warmup_frames:
                    # Keep counter-named fields that increased, are WIDE (uint32, so
                    # they don't wrap), and had no mid back-step in warmup.
                    self._watched = {i for i in self._candidates
                                     if self._saw_inc.get(i)
                                     and self._max_seen.get(i, 0) > self._min_max
                                     and not self._bad.get(i)}
                    names = [self.headers[i] for i in sorted(self._watched)]
                    print(f"[consistency_check] warmup done: watching "
                          f"{len(self._watched)} wide monotonic counters: {names}")
            else:
                for i in self._watched:                       # prime explicit windows
                    v = _to_num(low_level_data[i]) if i < n else None
                    if v is not None:
                        self._win[i].append(v)
            return

        # ── Per-sample outlier-below-window check ───────────────────────────
        violations = []
        for i in self._watched:
            v = _to_num(low_level_data[i]) if i < n else None
            if v is None:
                continue
            if self._violation(i, v):
                violations.append((self.headers[i], min(self._win[i]), v))
            self._win[i].append(v)
        self._latest_violations = violations

        # ── Fold into a labeled EX-0014.02 incident (AINOS3-25) ──────────────
        mode = _MODE_NAME.get(int(m), str(m)) if (m := (_to_num(low_level_data[self._mode_idx])
              if self._mode_idx is not None and self._mode_idx < n else None)) is not None else ""
        sub = f"{violations[0][0]}-backwards" if violations else ""
        closed = self._incident_agg.update(
            self._frame_count - 1, bool(violations),
            mode=mode, cluster="EX-0014.02", sub_technique=sub,
            confidence=1.0 if violations else 0.0)
        if closed is not None:
            self._write_incident(closed)

        if violations:
            for field, wmin, v in violations[:4]:
                print(f"[consistency_check][SPOOF] frame={self._frame_count} "
                      f"{field} = {v:.0f} is below its recent counter floor "
                      f"{wmin:.0f} (bus-spoof / corruption, EX-0014.02)")
        elif self._heartbeat_every > 0 and self._frame_count % self._heartbeat_every == 0:
            print(f"[consistency_check] frame={self._frame_count} watching "
                  f"{len(self._watched)} counters, clean")

        if self._side_file_path is not None:
            self._side_file_buffer.append([
                self._frame_count - 1,
                ";".join(f"{f}:{int(w)}>{int(v)}" for f, w, v in violations),
                int(bool(violations)),
            ])
            if len(self._side_file_buffer) >= self._side_file_flush_every:
                self._flush_side_file()

    def render_reasoning(self):
        return {
            "spoof_alert": bool(self._latest_violations),
            "violations": [f for f, _, _ in self._latest_violations],
            "n_incidents": self._n_incidents,
            "frame": self._frame_count,
        }

    def _write_incident(self, incident):
        """Append a closed spoof incident (same Incident row format as the other
        gates, so all three incident streams merge downstream)."""
        self._n_incidents += 1
        print(f"[consistency_check][INCIDENT] #{self._n_incidents} "
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
                w.writerow(["frame_idx", "violations", "spoof_alert"])
                self._side_file_header_written = True
            w.writerows(self._side_file_buffer)
        self._side_file_buffer = []
