# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
# Licensed under the NASA Open Source Agreement version 1.3
"""Isolation Forest Learner plugin (Tier 1).

Loads a pickled IsolationForest + feature schema produced by
`components/onair/training/train.py` and scores each incoming frame.

Supports two pickle layouts:

  - **Per-scenario** (`{"models": {scn: IF, ...}, "schema": {...}, ...}`):
    the plugin routes each frame through `models[scenario]` based on the
    `Scenario` config option, and uses a per-scenario threshold loaded
    from `<model>.calibration.json` (output of
    `components/onair/training/calibrate.py`). This is the v2/v3 layout.
  - **Single-model** (`{"model": IF, "schema": {...}, ...}`): legacy
    layout from the pre-per-scenario trainer; uses a single threshold.

Returns `{"anomaly_score": float, "is_anomaly": bool, "scenario": str,
"threshold": float}` from `render_reasoning()`.

`csv_output` runs as a knowledge_rep plugin so its `update()` never sees the
learner's `high_level_data` (vehicle_rep calls `construct.update(frame)` with
a single arg). To persist scores for offline analysis, this plugin writes a
sibling side-file `iforest_out_<timestamp>_pid<N>.csv` next to the
`csv_out_*.csv` rotations, joined by pid + cumulative row index. See
`components/onair/training/loader.py::attach_iforest_scores`.

Activation:
1. Train: `python3 components/onair/training/train.py --per-scenario ...`
2. Calibrate: `python3 components/onair/training/calibrate.py --model <pkl> --manifest <baselines>`
3. `nos3_security.ini`:
       LearnersPluginDict = {'iforest': 'cf/onair/plugins/isolation_forest/__init__.py'}

       [ISOLATION_FOREST]
       ModelPath = data/onair/models/iforest_per_scenario_v3_multiuptime.pkl
       CalibrationPath = data/onair/models/iforest_per_scenario_v3_multiuptime.calibration.json
       Scenario = nominal_ops
       AnomalyThreshold = 0.0   # fallback if calibration missing for the scenario
4. Sync: re-run the OnAIR build (CMakeLists copies plugins to fsw/build/exe/cpu1/cf/onair/plugins/)
   plus the model + calibration JSON into a path accessible at runtime cwd.
"""

from __future__ import annotations

import ast
import configparser
import csv
import json
import os
import pickle
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin


def _coerce_scalar(value: Any) -> float:
    """Convert a frame scalar (string from sbn_adapter `str()`, or already numeric) to float.

    Empty / `[0]` / `nan` placeholders return 0.0 to match training-time semantics.
    """
    if value in ("", "[0]", "nan", None):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_list_value(value: Any):
    """Coerce a frame entry to a Python list (or None).

    Handles both Python list (live frame from sbn_adapter) and string-serialized
    list (CSV-path or `[0]`/'' placeholders).
    """
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return None
    if value in ("[0]", "", "nan"):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, list) else None


def _extract_at_path(value, path: list[int]) -> float:
    """Walk `path` into a nested list and return the leaf as float; 0.0 if missing."""
    cur = value
    for i in path:
        if not isinstance(cur, list) or i >= len(cur):
            return 0.0
        cur = cur[i]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return 0.0


class Plugin(AIPlugin):
    """Isolation Forest anomaly scorer.

    State across frames: `prev_raw` (last raw feature vector) used to compute
    deltas. First frame after init has zero deltas (matches training-time
    file-boundary masking).
    """

    DEFAULTS = {
        "ModelPath": "data/onair/models/iforest_per_scenario_v3_multiuptime.pkl",
        "CalibrationPath": "",  # empty → derive as <model>.calibration.json
        "Scenario": "nominal_ops",  # which per-scenario IF to route through
        "AnomalyThreshold": "0.0",  # fallback if calibration missing
        # Side-file writer: persists per-frame scores so offline tooling can
        # join them to the sibling csv_out_*.csv. Default ON because the
        # alternative is "scores live only in stdout."
        "WriteSideFile": "true",
        # Default mirrors csv_output's OutputDir; both plugins resolve relative
        # to OnAIR cwd (fsw/build/exe/cpu1) at runtime.
        "SideFileOutputDir": "../../../../data/onair/csv",
        "SideFileFlushEvery": "10",  # rows buffered before each fsync-less append
        # AINOS3-121: golden-frame capture for the offline IF-audit harness.
        # When >0, every Nth frame the EXACT feature vector the model scored is
        # saved (with score + scenario) to an .npz, so an offline decision_function
        # can be checked against the live score and used for per-feature positive
        # controls. 0 = off (production default).
        "GoldenCaptureEvery": "0",
        "GoldenCaptureMax": "60",
        # Startup-transient suppression. Training uses --skip-warmup-rows 30
        # because the first ~30 frames after OnAIR connects have not-yet-
        # arrived MIDs (placeholder values + huge first-arrival deltas) that
        # the model treats as anomalous. During warmup we still record raw
        # score / is_anomaly to the side-file but never raise ALERT / CLEAR.
        "WarmupFrames": "30",
        # Hysteresis: require N consecutive anomaly (resp. nominal) frames
        # before the operational state transitions. Kills 1-frame spikes
        # like yesterday's "ALERT frame=N / CLEAR frame=N+1" pattern; the
        # 6s cmd-injection window (≈30 frames at 5 Hz) leaves plenty of
        # headroom for N=2. Raw is_anomaly in the side-file is unaffected.
        "AlertHysteresis": "2",
        "ClearHysteresis": "2",
        # Periodic threshold recalibration. v3 calibration was a point-in-
        # time snapshot; long-uptime stack drift pushes the operational FP
        # rate above target. When enabled, the plugin keeps a rolling
        # window of recent scores, and every RecalibrationIntervalFrames
        # frames recomputes the threshold as the Pth percentile of that
        # window (P = target_fp_rate from the calibration JSON). The new
        # threshold is persisted back to the calibration JSON atomically.
        # Off by default — opt-in via ini so the safety semantics (clamp,
        # alert-active gate) get reviewed before they touch operations.
        "RecalibrationEnabled": "false",
        # Window length. 9000 frames ≈ 30 min at the live ~5 Hz cadence,
        # long enough to dwarf any plausible attack duration so outlier
        # filtering by percentile is robust.
        "RecalibrationWindowFrames": "9000",
        # How often the plugin attempts recalibration once the window is
        # full. 1500 ≈ 5 min between attempts; bounds how fast the
        # threshold can crawl without flooding the calibration JSON.
        "RecalibrationIntervalFrames": "1500",
        # Per-attempt clamp on |threshold shift|, as a percentage of the
        # current threshold's magnitude. Prevents one bad recalibration
        # from blowing up the threshold; multiple attempts converge.
        "RecalibrationMaxDeltaPct": "50",
        # Runtime scenario routing. When enabled, every frame reads
        # RoutingSourceHeader (e.g. ADCS_GNC.Mode) and looks up which
        # per-scenario IF to score with via RoutingModeMap. Hysteresis
        # avoids thrashing across mode boundaries. OFF by default —
        # legacy static-Scenario behavior preserved byte-for-byte.
        "RuntimeRouting": "false",
        # When dry-run, the plugin logs the histogram of observed
        # RoutingSourceHeader values but does NOT switch models. Use it
        # to build RoutingModeMap from real telemetry before flipping the
        # apply path on; map values that aren't in the pickle's models
        # would otherwise silently route to the static Scenario fallback.
        "RoutingDryRun": "true",
        "RoutingSourceHeader": "ADCS_GNC.Mode",
        # JSON-encoded {mode_value_as_string: scenario_name}. Empty map
        # means "every value falls through to the static Scenario." The
        # actual mapping is mission-specific (ADCS mode enum semantics);
        # the dry-run pass produces the data needed to populate it.
        "RoutingModeMap": "{}",
        # Consecutive frames a new mode must persist before the model
        # switch fires. 30 frames ≈ 6 s at 5 Hz — matches the mode-
        # transition residency expected on a real mode change and
        # suppresses one-frame mode-bit jitter.
        "RoutingHysteresisFrames": "30",
        # In dry-run, emit the running histogram of observed mode values
        # every N frames so the operator sees what's happening without
        # tailing a per-frame log.
        "RoutingDryRunReportEvery": "1000",
        # Mode-switch warmup. After every routing-driven scenario switch
        # the plugin re-arms an N-frame warmup window during which ALERT
        # and CLEAR transitions are suppressed (raw is_anomaly still
        # written to the side-file). 600 frames ≈ 2 min at 5 Hz —
        # empirically covers the ~2-min FP transient observed on every
        # ADCS mode change (see project_v5_mode_soak_protocol). The
        # switch also clears prev_raw and the hysteresis counters so the
        # first new-mode frame has zero deltas (matches cold-start +
        # training file-boundary semantics) and stale anomaly/nominal
        # streaks don't carry across the boundary.
        "ModeSwitchWarmupFrames": "600",
    }

    def __init__(self, name, headers):
        super().__init__(name, headers)

        cfg = self._load_config()
        self.model_path = cfg.get("modelpath", self.DEFAULTS["ModelPath"])
        cal_path = cfg.get("calibrationpath", self.DEFAULTS["CalibrationPath"]).strip()
        if not cal_path:
            cal_path = self.model_path.removesuffix(".pkl") + ".calibration.json"
        self.calibration_path = cal_path
        self.scenario = cfg.get("scenario", self.DEFAULTS["Scenario"])
        # Immutable reference to the ini-configured scenario. Used by the
        # routing layer as the fail-safe target when an observed mode value
        # has no entry in RoutingModeMap — graceful degradation back to a
        # known, calibrated baseline rather than silently sticking on
        # whatever the last switched scenario happened to be.
        self._static_scenario = self.scenario
        fallback_threshold = float(cfg.get("anomalythreshold", self.DEFAULTS["AnomalyThreshold"]))

        with open(self.model_path, "rb") as f:
            artifact = pickle.load(f)

        self.schema = artifact["schema"]
        self.include_deltas = artifact.get("config", {}).get("include_deltas", True)

        # Per-scenario layout takes precedence; legacy single-model is fallback.
        # `self._models` holds every available scenario IF — runtime routing
        # needs to switch between them at frame granularity.
        if "models" in artifact:
            self._models: dict[str, Any] = dict(artifact["models"])
            if self.scenario not in self._models:
                raise ValueError(
                    f"Scenario {self.scenario!r} not in pickle; "
                    f"available: {sorted(self._models)}"
                )
            self.model = self._models[self.scenario]
            print(f"[iforest] per-scenario layout, routing through {self.scenario!r} IF "
                  f"({len(self._models)} models in pickle)")
        elif "model" in artifact:
            # Synthesize a single-entry models dict so the rest of the
            # plugin treats both layouts uniformly.
            self.model = artifact["model"]
            self._models = {self.scenario: self.model}
            print(f"[iforest] single-model layout, scenario={self.scenario!r} (advisory)")
        else:
            raise ValueError(f"{self.model_path}: no 'models' or 'model' key in pickle")

        # Calibration: per-scenario threshold from calibrate.py output. Falls
        # back to AnomalyThreshold when the file is missing or the scenario
        # entry is missing/null.
        self.threshold = self._load_threshold(self.scenario, fallback_threshold)

        # Build header index for fast frame -> feature mapping. Schema is
        # frozen at training time; column-type drift in scoring data can't
        # change feature count.
        self._header_to_idx = {h: i for i, h in enumerate(headers)}
        delta_only_set: set[str] = set(self.schema.get("delta_only_columns") or [])
        self._scalar_indices: list[int] = []
        scalar_delta_only_flags: list[bool] = []
        for c in self.schema["scalar_columns"]:
            if c not in self._header_to_idx:
                continue
            self._scalar_indices.append(self._header_to_idx[c])
            scalar_delta_only_flags.append(c in delta_only_set)
        self._list_columns: list[tuple[int, list[list[int]]]] = []
        list_delta_only_flags: list[bool] = []
        for col, layout in self.schema["list_columns"].items():
            if col not in self._header_to_idx:
                continue
            self._list_columns.append((self._header_to_idx[col], layout["paths"]))
            list_delta_only_flags.append(col in delta_only_set)

        self.n_raw = len(self._scalar_indices) + sum(len(p) for _, p in self._list_columns)
        # Mask over raw-vector indices marking columns whose raw values are
        # suppressed from the feature matrix (but whose deltas survive).
        # Must mirror the training-time logic in features.build_features.
        if delta_only_set:
            mask = np.zeros(self.n_raw, dtype=bool)
            for i, is_do in enumerate(scalar_delta_only_flags):
                mask[i] = is_do
            offset = len(self._scalar_indices)
            for (_, paths), is_do in zip(self._list_columns, list_delta_only_flags):
                if is_do:
                    mask[offset:offset + len(paths)] = True
                offset += len(paths)
            self._delta_only_raw_mask = mask
            self._raw_keep_mask = ~mask
        else:
            self._delta_only_raw_mask = None
            self._raw_keep_mask = None
        self.prev_raw: np.ndarray | None = None
        self._latest_score: float | None = None
        self._latest_is_anomaly: bool = False

        # Operational logging — heartbeat every N-th frame plus every
        # anomaly transition, so silence is distinguishable from "everything
        # nominal." Persisted scores go to the side-file (below).
        self._frame_count = 0
        self._heartbeat_every = int(cfg.get("heartbeatevery", "100"))

        # Warmup + hysteresis state.
        #   _alert_active tracks the *operational* alert state (post-
        #   warmup, post-hysteresis). _was_anomaly tracks the raw per-frame
        #   is_anomaly used for side-file alert/cleared edge bits.
        self._warmup_frames = int(cfg.get("warmupframes", self.DEFAULTS["WarmupFrames"]))
        self._alert_hyst = max(1, int(cfg.get("alerthysteresis", self.DEFAULTS["AlertHysteresis"])))
        self._clear_hyst = max(1, int(cfg.get("clearhysteresis", self.DEFAULTS["ClearHysteresis"])))
        self._consec_anomaly = 0
        self._consec_nominal = 0
        self._alert_active = False
        self._was_anomaly = False

        # Recalibration state. Off by default so the safety semantics are
        # explicit. The two parallel deques (scores + alert_active) let the
        # recalibrator skip windows that overlap a latched alert — without
        # that gate, an attack period would contaminate the nominal sample
        # and slide the threshold to permit similar attacks.
        self._recal_enabled = cfg.get(
            "recalibrationenabled", self.DEFAULTS["RecalibrationEnabled"]
        ).strip().lower() == "true"
        self._recal_window = max(1, int(cfg.get(
            "recalibrationwindowframes", self.DEFAULTS["RecalibrationWindowFrames"])))
        self._recal_interval = max(1, int(cfg.get(
            "recalibrationintervalframes", self.DEFAULTS["RecalibrationIntervalFrames"])))
        self._recal_max_delta_pct = float(cfg.get(
            "recalibrationmaxdeltapct", self.DEFAULTS["RecalibrationMaxDeltaPct"]))
        self._recal_scores: deque[float] = deque(maxlen=self._recal_window)
        self._recal_alert_in_window: deque[bool] = deque(maxlen=self._recal_window)
        # Parallel deque tagging each window frame with the active scenario.
        # A window that spans multiple scenarios (e.g., a mode switch fell
        # inside it) is treated as contaminated — the percentile would mix
        # distributions and the resulting threshold would be wrong for both.
        self._recal_scenarios: deque[str] = deque(maxlen=self._recal_window)
        self._recal_frames_since = 0
        self._recal_attempt_count = 0

        # Runtime scenario routing state.
        self._routing_enabled = cfg.get(
            "runtimerouting", self.DEFAULTS["RuntimeRouting"]
        ).strip().lower() == "true"
        self._routing_dry_run = cfg.get(
            "routingdryrun", self.DEFAULTS["RoutingDryRun"]
        ).strip().lower() == "true"
        self._routing_source = cfg.get(
            "routingsourceheader", self.DEFAULTS["RoutingSourceHeader"]
        ).strip()
        try:
            raw_map = cfg.get("routingmodemap", self.DEFAULTS["RoutingModeMap"])
            self._routing_map: dict[str, str] = {
                str(k): str(v) for k, v in json.loads(raw_map).items()
            }
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"[iforest][route] WARNING: RoutingModeMap parse failed ({e!r}); "
                  f"treating as empty map")
            self._routing_map = {}
        self._routing_hyst = max(1, int(cfg.get(
            "routinghysteresisframes", self.DEFAULTS["RoutingHysteresisFrames"])))
        self._routing_report_every = max(1, int(cfg.get(
            "routingdryrunreportevery", self.DEFAULTS["RoutingDryRunReportEvery"])))
        self._routing_source_idx: int | None = (
            self._header_to_idx.get(self._routing_source) if self._routing_enabled else None
        )
        # Bad config is loud at startup: map values must name actual scenarios.
        if self._routing_enabled:
            unknown = [v for v in self._routing_map.values() if v not in self._models]
            if unknown:
                raise ValueError(
                    f"RoutingModeMap targets scenarios not in pickle: {sorted(set(unknown))}; "
                    f"available: {sorted(self._models)}"
                )
            if self._routing_source_idx is None:
                print(f"[iforest][route] WARNING: RoutingSourceHeader "
                      f"{self._routing_source!r} not found in headers; "
                      f"routing inactive for this session")
        self._routing_observed: Counter[str] = Counter()
        self._routing_pending_scenario: str | None = None
        self._routing_pending_streak = 0
        self._routing_switches = 0
        # Mode-switch warmup: counter set by _switch_scenario(), counted
        # down each frame in update(). While > 0 the frame is treated as
        # warmup (alert/clear suppressed, hysteresis counters held at 0).
        self._mode_switch_warmup_frames = max(0, int(cfg.get(
            "modeswitchwarmupframes", self.DEFAULTS["ModeSwitchWarmupFrames"])))
        self._mode_switch_warmup_remaining = 0

        # Side-file writer setup.
        self._side_file_path: str | None = None
        self._side_file_buffer: list[list[Any]] = []
        self._side_file_header_written = False
        self._side_file_flush_every = int(
            cfg.get("sidefileflushevery", self.DEFAULTS["SideFileFlushEvery"])
        )
        write_side = cfg.get(
            "writesidefile", self.DEFAULTS["WriteSideFile"]
        ).strip().lower() == "true"
        if write_side:
            side_dir = cfg.get(
                "sidefileoutputdir", self.DEFAULTS["SideFileOutputDir"]
            )
            os.makedirs(side_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
            self._side_file_path = os.path.join(
                side_dir, f"iforest_out_{ts}_pid{os.getpid()}.csv"
            )

        # AINOS3-121 golden capture
        self._golden_every = int(cfg.get("goldencaptureevery",
                                         self.DEFAULTS["GoldenCaptureEvery"]))
        self._golden_max = int(cfg.get("goldencapturemax",
                                       self.DEFAULTS["GoldenCaptureMax"]))
        self._golden_rows: list[tuple] = []
        self._golden_path: str | None = None
        if self._golden_every > 0:
            side_dir = cfg.get("sidefileoutputdir", self.DEFAULTS["SideFileOutputDir"])
            os.makedirs(side_dir, exist_ok=True)
            gts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
            self._golden_path = os.path.join(
                side_dir, f"iforest_golden_{gts}_pid{os.getpid()}.npz")
            print(f"[iforest][golden] capturing every {self._golden_every} frames "
                  f"(max {self._golden_max}) -> {self._golden_path}")

        print(f"[iforest] threshold={self.threshold:+.4f} "
              f"(score < threshold ⇒ anomaly), n_raw_features={self.n_raw}, "
              f"heartbeat every {self._heartbeat_every} frames")
        print(f"[iforest] warmup={self._warmup_frames} frames, "
              f"hysteresis alert/clear={self._alert_hyst}/{self._clear_hyst}")
        if self._recal_enabled:
            tfp = (f"{self._target_fp_rate*100:.2f}%"
                   if self._target_fp_rate is not None else "MISSING")
            print(f"[iforest] recalibration ENABLED: window={self._recal_window} "
                  f"frames, interval={self._recal_interval} frames, "
                  f"clamp=±{self._recal_max_delta_pct:.0f}%, target FP={tfp}")
        if self._routing_enabled:
            mode = "DRY-RUN" if self._routing_dry_run else "APPLY"
            map_str = (f"{len(self._routing_map)} entries"
                       if self._routing_map else "EMPTY")
            print(f"[iforest] routing ENABLED ({mode}): source={self._routing_source}, "
                  f"map={map_str}, hysteresis={self._routing_hyst} frames")
            print(f"[iforest] mode-switch warmup={self._mode_switch_warmup_frames} "
                  f"frames (re-armed on every scenario switch)")
        if self._side_file_path is not None:
            print(f"[iforest] side-file → {self._side_file_path} "
                  f"(flush every {self._side_file_flush_every} rows)")
        else:
            print("[iforest] side-file writer disabled")

    @staticmethod
    def _load_config() -> dict:
        ini_path = os.environ.get("ONAIR_INI_FILE") or "cf/onair/nos3_security.ini"
        if not os.path.exists(ini_path):
            return {}
        parser = configparser.ConfigParser()
        parser.read(ini_path)
        if parser.has_section("ISOLATION_FOREST"):
            return dict(parser.items("ISOLATION_FOREST"))
        return {}

    def _load_threshold(self, scenario: str, fallback: float) -> float:
        self._target_fp_rate: float | None = None
        if not os.path.exists(self.calibration_path):
            print(f"[iforest] WARNING: calibration file not found at "
                  f"{self.calibration_path}; using AnomalyThreshold fallback "
                  f"{fallback:+.4f}")
            return fallback
        try:
            with open(self.calibration_path) as f:
                cal = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[iforest] WARNING: could not read calibration {self.calibration_path}: {e}; "
                  f"using AnomalyThreshold fallback {fallback:+.4f}")
            return fallback
        thr = (cal.get("thresholds") or {}).get(scenario)
        if thr is None:
            print(f"[iforest] WARNING: no calibrated threshold for scenario "
                  f"{scenario!r} in {self.calibration_path}; using "
                  f"AnomalyThreshold fallback {fallback:+.4f}")
            return fallback
        target = cal.get("target_fp_rate")
        if isinstance(target, (int, float)):
            self._target_fp_rate = float(target)
        target_str = f" (calibrated @ FP={target*100:.1f}%)" if isinstance(target, (int, float)) else ""
        print(f"[iforest] loaded calibrated threshold for {scenario!r}: "
              f"{thr:+.4f}{target_str}")
        return float(thr)

    def _frame_to_raw(self, frame) -> np.ndarray:
        """Build the raw (pre-delta) feature vector from one telemetry frame."""
        raw = np.zeros(self.n_raw, dtype=np.float64)

        # Scalar columns
        for out_i, frame_i in enumerate(self._scalar_indices):
            raw[out_i] = _coerce_scalar(frame[frame_i])

        # List columns: extract scalars at the canonical leaf-paths recorded at
        # training time. Missing/short structures yield 0.0 (matches training).
        offset = len(self._scalar_indices)
        for frame_i, paths in self._list_columns:
            parsed = _parse_list_value(frame[frame_i])
            if parsed is not None:
                for j, path in enumerate(paths):
                    raw[offset + j] = _extract_at_path(parsed, path)
            offset += len(paths)

        return raw

    def _read_threshold_from_json(self, scenario: str, fallback: float) -> float:
        """Quiet variant of `_load_threshold` for runtime scenario switches.

        No init-style logging — just returns the calibrated threshold for
        the named scenario, falling back when the file or entry is missing.
        Used by `_switch_scenario` so a routing-driven model swap picks up
        any recalibration-driven threshold updates without flooding stdout.
        """
        if not os.path.exists(self.calibration_path):
            return fallback
        try:
            with open(self.calibration_path) as f:
                cal = json.load(f)
        except (OSError, json.JSONDecodeError):
            return fallback
        thr = (cal.get("thresholds") or {}).get(scenario)
        return float(thr) if thr is not None else fallback

    def _handle_routing(self, frame) -> None:
        """Observe the routing source value and (unless dry-run) drive the
        per-scenario model switch via N-frame hysteresis.

        Dry-run mode skips the switch path entirely and just accumulates
        a histogram of observed mode values, periodically logged so the
        operator can build the RoutingModeMap from real telemetry before
        flipping the apply path on.
        """
        if self._routing_source_idx is None:
            return
        raw_value = frame[self._routing_source_idx]
        # ADCS mode is an int but the frame entry arrives as a string from
        # the SBN adapter; normalize to string so the map lookup is robust.
        key = str(raw_value).strip()
        self._routing_observed[key] += 1

        if self._routing_dry_run:
            if (self._routing_report_every > 0
                    and self._frame_count > 0
                    and self._frame_count % self._routing_report_every == 0):
                self._log_dry_run_histogram()
            return

        target = self._routing_map.get(key, self._static_scenario)
        if target == self.scenario:
            self._routing_pending_scenario = None
            self._routing_pending_streak = 0
            return
        if self._routing_pending_scenario != target:
            self._routing_pending_scenario = target
            self._routing_pending_streak = 1
        else:
            self._routing_pending_streak += 1
        if self._routing_pending_streak >= self._routing_hyst:
            self._switch_scenario(target)
            self._routing_pending_scenario = None
            self._routing_pending_streak = 0

    def _log_dry_run_histogram(self) -> None:
        n = sum(self._routing_observed.values())
        in_map = sum(c for k, c in self._routing_observed.items()
                     if k in self._routing_map)
        top = ", ".join(f"{k}:{c}"
                        for k, c in self._routing_observed.most_common(8))
        coverage = f"{in_map}/{n}" if self._routing_map else "no map yet"
        print(f"[iforest][route][dry-run] frame={self._frame_count} "
              f"observed N={n} (in-map: {coverage}) top: {top}")

    def _switch_scenario(self, new_scenario: str) -> None:
        if new_scenario == self.scenario:
            return
        if new_scenario not in self._models:
            print(f"[iforest][route] WARNING: target scenario "
                  f"{new_scenario!r} not in pickle; staying on "
                  f"{self.scenario!r}")
            return
        old_scn = self.scenario
        old_thr = self.threshold
        self.scenario = new_scenario
        self.model = self._models[new_scenario]
        self.threshold = self._read_threshold_from_json(
            new_scenario, fallback=self.threshold)
        self._routing_switches += 1
        # Re-arm mode-switch warmup + clear cross-mode delta and stale
        # hysteresis streaks. prev_raw from the old mode would otherwise
        # produce a huge cross-mode delta on the first new-mode frame
        # that the new IF has never seen.
        self.prev_raw = None
        self._consec_anomaly = 0
        self._consec_nominal = 0
        self._mode_switch_warmup_remaining = self._mode_switch_warmup_frames
        warmup_tag = (f", warmup {self._mode_switch_warmup_frames} frames"
                      if self._mode_switch_warmup_frames > 0 else "")
        print(f"[iforest][route] scenario {old_scn} → {new_scenario} "
              f"at frame={self._frame_count + 1}; "
              f"threshold {old_thr:+.5f} → {self.threshold:+.5f}"
              f"{warmup_tag}")

    def update(self, low_level_data=None, high_level_data=None):
        if not low_level_data:
            return
        if self._routing_enabled:
            self._handle_routing(low_level_data)
        raw = self._frame_to_raw(low_level_data)

        if self.include_deltas:
            if self.prev_raw is None:
                delta = np.zeros_like(raw)
            else:
                delta = raw - self.prev_raw
            if self._raw_keep_mask is not None:
                features = np.concatenate([raw[self._raw_keep_mask], delta])
            else:
                features = np.concatenate([raw, delta])
        else:
            features = raw

        self.prev_raw = raw
        score = float(self.model.decision_function(features.reshape(1, -1))[0])
        is_anomaly = score < self.threshold
        self._latest_score = score
        self._latest_is_anomaly = is_anomaly
        self._frame_count += 1

        # Operational alert state machine: warmup suppresses transitions
        # entirely AND keeps the hysteresis counters at zero, so the first
        # post-warmup anomaly frame is counted as `consec=1` rather than
        # inheriting a tall warmup-era streak that would fire ALERT on the
        # very next frame. alert/cleared edge bits and the [ALERT]/[CLEAR]
        # log lines reflect the *operational* state, not the raw per-frame
        # is_anomaly (which is preserved verbatim in the side-file for
        # offline analysis).
        in_cold_warmup = self._frame_count <= self._warmup_frames
        in_mode_warmup = self._mode_switch_warmup_remaining > 0
        in_warmup = in_cold_warmup or in_mode_warmup
        if in_mode_warmup:
            self._mode_switch_warmup_remaining -= 1
        alert = False
        cleared = False
        if not in_warmup:
            if is_anomaly:
                self._consec_anomaly += 1
                self._consec_nominal = 0
            else:
                self._consec_nominal += 1
                self._consec_anomaly = 0
            if not self._alert_active and self._consec_anomaly >= self._alert_hyst:
                self._alert_active = True
                alert = True
            elif self._alert_active and self._consec_nominal >= self._clear_hyst:
                self._alert_active = False
                cleared = True

        # Edge-triggered alert on transitions, plus a heartbeat so silence
        # is distinguishable from "plugin alive but everything nominal".
        if alert:
            print(f"[iforest][ALERT] frame={self._frame_count} "
                  f"scenario={self.scenario} score={score:+.4f} "
                  f"threshold={self.threshold:+.4f} "
                  f"(entered anomaly after {self._alert_hyst} consec frames)")
        elif cleared:
            print(f"[iforest][CLEAR] frame={self._frame_count} "
                  f"scenario={self.scenario} score={score:+.4f} "
                  f"(left anomaly after {self._clear_hyst} consec nominal frames)")
        elif self._heartbeat_every > 0 and self._frame_count % self._heartbeat_every == 0:
            tag = " [warmup]" if in_warmup else ""
            print(f"[iforest] frame={self._frame_count}{tag} "
                  f"scenario={self.scenario} score={score:+.4f} "
                  f"is_anomaly={is_anomaly} alert_active={self._alert_active}")
        self._was_anomaly = is_anomaly

        if self._side_file_path is not None:
            # frame_idx is 0-indexed so it lines up with csv_output's __row_idx
            # cumulative across same-pid rotations; printed log frames remain
            # 1-indexed (`self._frame_count`) for human readability. is_anomaly
            # is the raw decision; alert/cleared encode the operational
            # (warmup+hysteresis-gated) transition for the offline record.
            self._side_file_buffer.append([
                self._frame_count - 1,
                self.scenario,
                f"{score:.6f}",
                f"{self.threshold:.6f}",
                int(is_anomaly),
                int(alert),
                int(cleared),
            ])
            if len(self._side_file_buffer) >= self._side_file_flush_every:
                self._flush_side_file()

        # AINOS3-121: capture the exact scored feature vector for offline replay.
        if (self._golden_every > 0 and len(self._golden_rows) < self._golden_max
                and self._frame_count % self._golden_every == 0):
            self._golden_rows.append(
                (self._frame_count - 1, self.scenario, score, self.threshold,
                 int(is_anomaly), features.astype(np.float64).copy()))
            # flush incrementally (savez overwrites) so the file is always current,
            # e.g. while cycling modes; and once more on reaching the cap.
            if len(self._golden_rows) % 20 == 0 or len(self._golden_rows) >= self._golden_max:
                self._save_golden()

        if self._recal_enabled:
            self._recal_scores.append(score)
            self._recal_alert_in_window.append(self._alert_active)
            self._recal_scenarios.append(self.scenario)
            self._recal_frames_since += 1
            if (self._recal_frames_since >= self._recal_interval
                    and len(self._recal_scores) >= self._recal_window):
                self._try_recalibrate()
                self._recal_frames_since = 0

    def _save_golden(self) -> None:
        """Persist captured golden frames (feature vector + score) to .npz for the
        AINOS3-121 offline audit. feature_names gives the column->feature mapping."""
        if not self._golden_rows or self._golden_path is None:
            return
        try:
            np.savez(
                self._golden_path,
                frame_idx=np.array([r[0] for r in self._golden_rows], dtype=np.int64),
                scenario=np.array([r[1] for r in self._golden_rows]),
                score=np.array([r[2] for r in self._golden_rows], dtype=np.float64),
                threshold=np.array([r[3] for r in self._golden_rows], dtype=np.float64),
                is_anomaly=np.array([r[4] for r in self._golden_rows], dtype=np.int64),
                features=np.vstack([r[5] for r in self._golden_rows]),
                feature_names=np.array(self.schema["feature_names"]),
            )
            print(f"[iforest][golden] saved {len(self._golden_rows)} frames "
                  f"-> {self._golden_path}")
        except Exception as e:  # never let capture break scoring
            print(f"[iforest][golden] save failed: {e}")

    def _try_recalibrate(self) -> None:
        """Recompute the threshold from the rolling nominal window, if safe.

        Skipped when: target_fp_rate is missing from the calibration JSON,
        or any frame in the window had `alert_active=True` (the window
        overlaps a latched alarm — counts the alarm scores as nominal,
        which would slide the threshold to permit the very condition that
        raised it). Otherwise picks the Pth percentile as the new
        threshold and clamps the per-attempt shift to MaxDeltaPct.
        """
        self._recal_attempt_count += 1
        if self._target_fp_rate is None:
            print(f"[iforest][recal] attempt {self._recal_attempt_count} skipped: "
                  f"target_fp_rate missing from calibration JSON")
            return
        if any(self._recal_alert_in_window):
            n_alerted = sum(1 for a in self._recal_alert_in_window if a)
            print(f"[iforest][recal] attempt {self._recal_attempt_count} skipped: "
                  f"alert_active was True on {n_alerted}/{len(self._recal_alert_in_window)} "
                  f"window frames")
            return
        unique_scenarios = set(self._recal_scenarios)
        if len(unique_scenarios) > 1:
            # Window spans a scenario switch — mixing distributions would
            # give a percentile that's wrong for both scenarios. Wait for a
            # window that lives entirely under one scenario.
            print(f"[iforest][recal] attempt {self._recal_attempt_count} skipped: "
                  f"window spans {len(unique_scenarios)} scenarios "
                  f"({sorted(unique_scenarios)})")
            return

        pct = self._target_fp_rate * 100.0
        scores_arr = np.fromiter(self._recal_scores, dtype=np.float64)
        proposed = float(np.percentile(scores_arr, pct))

        # Clamp per-attempt shift. Use a small floor when the current
        # threshold is near zero — otherwise the clamp collapses and the
        # threshold cannot move off zero in any single attempt.
        base = max(abs(self.threshold), 0.01)
        max_delta = base * (self._recal_max_delta_pct / 100.0)
        delta = proposed - self.threshold
        if abs(delta) > max_delta:
            clamped = self.threshold + max_delta * (1.0 if delta > 0 else -1.0)
            print(f"[iforest][recal] attempt {self._recal_attempt_count}: "
                  f"Δ={delta:+.5f} exceeds clamp ±{max_delta:.5f}; "
                  f"applying clamped Δ={clamped - self.threshold:+.5f}")
            proposed = clamped

        old = self.threshold
        self.threshold = proposed
        persisted = self._persist_recalibration()
        print(f"[iforest][recal] attempt {self._recal_attempt_count}: "
              f"threshold {old:+.5f} → {self.threshold:+.5f} "
              f"(window={len(self._recal_scores)} frames, "
              f"target FP={pct:.2f}%, persisted={persisted})")

    def _persist_recalibration(self) -> bool:
        """Atomically write the updated threshold back to the calibration JSON.

        Reads-modifies-writes with a tmp + os.replace so a partial write
        cannot leave the JSON malformed. Records when and which scenario
        was recalibrated for offline auditing.
        """
        if not self.calibration_path:
            return False
        try:
            if os.path.exists(self.calibration_path):
                with open(self.calibration_path) as f:
                    cal = json.load(f)
            else:
                cal = {}
        except (OSError, json.JSONDecodeError) as e:
            print(f"[iforest][recal] WARNING: could not read calibration "
                  f"{self.calibration_path}: {e}")
            return False
        cal.setdefault("thresholds", {})[self.scenario] = float(self.threshold)
        cal["last_recalibration_utc"] = datetime.now(tz=timezone.utc).isoformat()
        cal["last_recalibration_scenario"] = self.scenario
        cal["last_recalibration_window_frames"] = len(self._recal_scores)
        tmp_path = self.calibration_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(cal, f, indent=2)
            os.replace(tmp_path, self.calibration_path)
        except OSError as e:
            print(f"[iforest][recal] WARNING: could not write calibration "
                  f"{self.calibration_path}: {e}")
            return False
        return True

    def _flush_side_file(self) -> None:
        """Append buffered side-file rows; first flush emits the header."""
        if self._side_file_path is None or not self._side_file_buffer:
            return
        write_header = not self._side_file_header_written
        with open(self._side_file_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "frame_idx", "scenario", "score", "threshold",
                    "is_anomaly", "alert", "cleared",
                ])
                self._side_file_header_written = True
            w.writerows(self._side_file_buffer)
        self._side_file_buffer.clear()

    def render_reasoning(self):
        if self._latest_score is None:
            return {
                "anomaly_score": 0.0, "is_anomaly": False,
                "scenario": self.scenario, "threshold": self.threshold,
            }
        return {
            "anomaly_score": self._latest_score,
            "is_anomaly": self._latest_is_anomaly,
            "scenario": self.scenario,
            "threshold": self.threshold,
        }
