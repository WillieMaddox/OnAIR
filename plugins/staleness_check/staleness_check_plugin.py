# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""Staleness-check learner — the FOURTH detector gate, for telemetry-denial /
frozen-stream attacks.

The EX-0012.02 validation (2026-07-16) established a class both the dynamics-IF,
the rule-gate, AND the consistency-check all miss: an attacker uses CFE_SB
DISABLE_ROUTE to sever a `MsgId → pipe` route, so that MID stops reaching a
subscriber. The MID's telemetry FREEZES. A frozen stream has no forward delta, so
the IF (constant input), rule-gate (no flag-drop/spike), and consistency-check (no
backwards step) all miss it. Same family as EX-0014.03 frozen fields.

THE SUBTLE PART is OnAIR's double buffer. A frozen field does NOT read as a single
constant value — the two buffers hold the last two received values, so OnAIR
alternates between them: a frozen counter reads e.g. 70227↔70243 every frame,
never constant. (That is why a naive "value unchanged for N frames" check never
fires on a real freeze.) The invariant that DOES hold: a monotonic counter's
running MAX stops advancing. When the MID is LIVE the max climbs every frame or
two (even through the flicker); when it FREEZES the max is pinned at the last
received value. So the primitive is **the counter's max has not advanced for N
frames**.

Watched counters are WIDE monotonic counters (uint32, max > MinCounterMax) — the
same non-wrapping class the consistency-check uses, because a uint8 counter wraps
(255→0) which also stalls the running max. Event/error tallies (`*ErrorCount`,
`Unexpected*`, `Filtered*`, …) are excluded: they advance sporadically, so a long
no-advance run is normal. Discovery runs AFTER a settle period (MIDs come online
over the first frames) and sets a per-counter threshold scaled to that counter's
own steady-state no-advance gap, so a slow-but-live counter (e.g. EVS every ~14
frames) never false-alarms while a fast one is caught quickly.

Emits a sibling side-file `staleness_out_<ts>_pid<N>.csv` and, via the shared
IncidentAggregator (AINOS3-25), labeled EX-0012.02 incidents spanning the freeze.
Operators OR this incident stream with the other three gates.
"""
from __future__ import annotations

import configparser
import csv
import datetime
import os
import re
import sys as _sys

from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin

# Reuse the incident aggregator (AINOS3-25) from the sibling xgb_classifier package.
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xgb_classifier"))
from incident import IncidentAggregator  # noqa: E402


_MODE_NAME = {0: "PASSIVE", 1: "BDOT", 2: "SUNSAFE", 3: "INERTIAL"}

# Liveness counters carry a count/packet token in the name.
_COUNTER_NAME = re.compile(r"(Count|Counter|Packets)", re.IGNORECASE)

# Error/event tallies advance sporadically (on an error / dropped / unexpected
# event), so a long no-advance run is NORMAL — they are not liveness indicators.
_EVENT_NAME = re.compile(
    r"(Error|Unexpected|Missed|Overflow|NoSubscribers|Fail|Filtered|Squelch|"
    r"Invalid|Dropped|Trunc|Duplicate)", re.IGNORECASE)


def _to_num(value):
    """Parse a telemetry cell to float; None for the `[0]` init sentinel / text."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Plugin(AIPlugin):
    """Per-MID liveness / staleness detector for telemetry-denial (EX-0012.02)."""

    DEFAULTS = {
        "WriteSideFile": "true",
        "SideFileOutputDir": "../../../../data/csv",
        "SideFileFlushEvery": "10",
        # Skip the startup transient BEFORE measuring liveness (MIDs come online
        # over several frames).
        "SettleFrames": "40",
        # Long discovery window: OnAIR polls FASTER than the MIDs publish, so a live
        # wide counter's max advances only every ~14-27 frames — a short window sees
        # too few advances. A long window also makes the AVERAGE advance interval
        # (below) a stable estimate.
        "WarmupFrames": "250",
        "LivenessFields": "auto",
        # Discovery uses the AVERAGE advance interval (window / advances), NOT the max
        # no-advance run. The max is an extreme-value statistic (dominated by the one
        # longest pause) and is high-variance run-to-run — that made the watched set
        # and thresholds unstable (a counter watched offline but not live). The mean
        # interval is a stable count-based statistic. A counter is watched if it is
        # WIDE (max > MinCounterMax, so it does not wrap), advanced >= MinAdvances
        # times, and its avg interval <= MaxAvgGap.
        "MinAdvances": "6",
        "MaxAvgGap": "30",
        "MinCounterMax": "256",
        # Per-counter stale threshold = max(StaleThreshold, StaleMarginK * avg
        # interval). K is large so the threshold clears the MAX nominal gap (~1.5-2x
        # the mean) with wide margin. Detection is intentionally high-latency — a
        # telemetry denial is persistent, so a late-but-certain catch beats a false
        # alarm.
        "StaleThreshold": "50",
        "StaleMarginK": "5",
        "HeartbeatEvery": "1000",
        "WriteIncidentFile": "true",
        "IncidentAlertHysteresis": "1",
        "IncidentClearHysteresis": "2",
        "IncidentMinAnomalyFrames": "1",
    }

    def __init__(self, name, headers):
        super().__init__(name, headers)
        cfg = self._load_config()

        def _int(k):
            return int(cfg.get(k.lower(), self.DEFAULTS[k]))

        self._settle_frames = _int("SettleFrames")
        self._warmup_frames = _int("WarmupFrames")
        self._discovery_end = self._settle_frames + self._warmup_frames
        self._min_advances = _int("MinAdvances")
        self._max_avg_gap = float(cfg.get("maxavggap", self.DEFAULTS["MaxAvgGap"]))
        self._min_max = float(cfg.get("mincountermax", self.DEFAULTS["MinCounterMax"]))
        self._stale_thresh = _int("StaleThreshold")
        self._margin_k = _int("StaleMarginK")
        self._heartbeat_every = _int("HeartbeatEvery")

        lf = cfg.get("livenessfields", self.DEFAULTS["LivenessFields"]).strip()
        idx = {h: i for i, h in enumerate(self.headers)}
        self._auto = (lf.lower() == "auto")
        if self._auto:
            self._candidates = {i for i, h in enumerate(self.headers)
                                if _COUNTER_NAME.search(h) and not _EVENT_NAME.search(h)}
            self._watched = set()
        else:
            self._candidates = set()
            self._watched = {idx[f.strip()] for f in lf.split(",")
                             if f.strip() and f.strip() in idx}

        self._prev_max = {}      # index → running max value
        self._notadv = {}        # index → consecutive frames the max has not advanced
        self._advances = {}      # discovery: how many times the max advanced
        self._max_val = {}       # discovery: largest value seen (wide-counter test)
        self._thr = {}           # per-counter stale threshold
        self._frame_count = 0
        self._latest_stale = []

        self._incident_agg = IncidentAggregator(
            alert_hysteresis=_int("IncidentAlertHysteresis"),
            clear_hysteresis=_int("IncidentClearHysteresis"),
            min_anomaly_frames=_int("IncidentMinAnomalyFrames"))
        self._incident_file_path = None
        self._incident_header_written = False
        self._n_incidents = 0
        self._mode_idx = idx.get("ADCS_GNC.Mode")

        self._side_file_path = None
        self._side_file_buffer = []
        self._side_file_header_written = False
        self._side_file_flush_every = _int("SideFileFlushEvery")
        out_dir = cfg.get("sidefileoutputdir", self.DEFAULTS["SideFileOutputDir"])
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
        if cfg.get("writesidefile", self.DEFAULTS["WriteSideFile"]).strip().lower() == "true":
            os.makedirs(out_dir, exist_ok=True)
            self._side_file_path = os.path.join(out_dir, f"staleness_out_{ts}_pid{os.getpid()}.csv")
        if cfg.get("writeincidentfile", self.DEFAULTS["WriteIncidentFile"]).strip().lower() == "true":
            os.makedirs(out_dir, exist_ok=True)
            self._incident_file_path = os.path.join(out_dir, f"staleness_incident_{ts}_pid{os.getpid()}.csv")

        note = (f"{len(self._candidates)} counter candidates" if self._auto
                else f"{len(self._watched)} forced")
        print(f"[staleness_check] max-advance liveness check ({note}); stale = max not "
              f"advanced for >={self._stale_thresh}(xK) frames; wide(max>{self._min_max:.0f}); "
              f"settle={self._settle_frames}, window={self._warmup_frames}")
        if self._side_file_path is not None:
            print(f"[staleness_check] side-file → {self._side_file_path}")
        if self._incident_file_path is not None:
            print(f"[staleness_check] incident-file → {self._incident_file_path}")

    @staticmethod
    def _load_config():
        ini_path = os.environ.get("ONAIR_INI_FILE") or "cf/onair/nos3_security.ini"
        if not os.path.exists(ini_path):
            return {}
        parser = configparser.ConfigParser()
        parser.read(ini_path)
        return dict(parser.items("STALENESS_CHECK")) if parser.has_section("STALENESS_CHECK") else {}

    def _step(self, i, v, discovery):
        """Advance the running-max bookkeeping for column i with value v.
        Returns the current no-advance run length."""
        pm = self._prev_max.get(i)
        if pm is None or v > pm:
            self._prev_max[i] = v
            self._notadv[i] = 0
            if discovery and pm is not None:
                self._advances[i] = self._advances.get(i, 0) + 1
        elif pm > 0 and (pm - v) > 0.5 * pm:
            # A large RELATIVE backwards drop = a counter WRAP (uint16 65535→0) or a
            # reset — the counter is alive (it kept counting past its type max), NOT
            # frozen. Re-baseline the running max to the post-wrap value. Without this
            # a wrapping wide counter (e.g. uint16 DS.FileWriteCounter) pins the max at
            # ~65515 for its whole next 0→65515 climb → a persistent false stale.
            # (A freeze holds the value ~constant, a small/zero drop — not a wrap.)
            self._prev_max[i] = v
            self._notadv[i] = 0
        else:
            self._notadv[i] = self._notadv.get(i, 0) + 1
        if discovery:
            self._max_val[i] = max(self._max_val.get(i, v), v)
        return self._notadv[i]

    def update(self, low_level_data=[], high_level_data={}):
        self._frame_count += 1
        n = len(low_level_data)

        # Phase 1 — settle: skip the startup transient.
        if self._frame_count <= self._settle_frames:
            return

        # Phase 2 — discovery: learn which wide counters are live + their cadence.
        if self._frame_count <= self._discovery_end:
            targets = self._candidates if self._auto else self._watched
            for i in targets:
                v = _to_num(low_level_data[i]) if i < n else None
                if v is not None:
                    self._step(i, v, discovery=True)
            if self._auto and self._frame_count == self._discovery_end:
                def _avg_gap(i):
                    a = self._advances.get(i, 0)
                    return (self._warmup_frames / a) if a else 1e9
                self._watched = {i for i in self._candidates
                                 if self._advances.get(i, 0) >= self._min_advances
                                 and _avg_gap(i) <= self._max_avg_gap
                                 and self._max_val.get(i, 0) > self._min_max}
                self._thr = {i: int(max(self._stale_thresh, self._margin_k * _avg_gap(i)))
                             for i in self._watched}
                self._notadv = {}   # reset for runtime
                names = [self.headers[i] for i in sorted(self._watched)]
                print(f"[staleness_check] discovery done: watching {len(self._watched)} "
                      f"wide liveness counters, e.g. {names[:10]}")
            return

        # Phase 3 — runtime: flag a counter whose max has not advanced past threshold.
        stale = []
        for i in self._watched:
            v = _to_num(low_level_data[i]) if i < n else None
            if v is None:
                continue
            run = self._step(i, v, discovery=False)
            thr = self._thr.get(i, self._stale_thresh)
            if run >= thr:
                stale.append((self.headers[i], run, thr))
        self._latest_stale = stale

        mode = _MODE_NAME.get(int(m), str(m)) if (m := (_to_num(low_level_data[self._mode_idx])
              if self._mode_idx is not None and self._mode_idx < n else None)) is not None else ""
        sub = f"{stale[0][0]}-stale" if stale else ""
        closed = self._incident_agg.update(
            self._frame_count - 1, bool(stale),
            mode=mode, cluster="EX-0012.02", sub_technique=sub,
            confidence=1.0 if stale else 0.0)
        if closed is not None:
            self._write_incident(closed)

        if stale:
            for field, run, thr in stale[:4]:
                if run == thr:   # log once at the onset
                    print(f"[staleness_check][STALE] frame={self._frame_count} "
                          f"{field} max has not advanced for {run} frames (>= {thr}) — "
                          f"telemetry frozen (route-disable / denial, EX-0012.02)")
        elif self._heartbeat_every > 0 and self._frame_count % self._heartbeat_every == 0:
            print(f"[staleness_check] frame={self._frame_count} watching "
                  f"{len(self._watched)} liveness counters, all live")

        if self._side_file_path is not None:
            self._side_file_buffer.append([
                self._frame_count - 1,
                ";".join(f"{f}:{r}" for f, r, _ in stale),
                int(bool(stale)),
            ])
            if len(self._side_file_buffer) >= self._side_file_flush_every:
                self._flush_side_file()

    def render_reasoning(self):
        return {
            "staleness_alert": bool(self._latest_stale),
            "stale_fields": [f for f, _, _ in self._latest_stale],
            "n_incidents": self._n_incidents,
            "frame": self._frame_count,
        }

    def _write_incident(self, incident):
        self._n_incidents += 1
        print(f"[staleness_check][INCIDENT] #{self._n_incidents} "
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
                w.writerow(["frame_idx", "stale_fields", "staleness_alert"])
                self._side_file_header_written = True
            w.writerows(self._side_file_buffer)
        self._side_file_buffer = []
