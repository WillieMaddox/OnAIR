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
"threshold": float}` from `render_reasoning()`. The csv_output plugin
appends these as columns automatically.

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
import json
import os
import pickle
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
        fallback_threshold = float(cfg.get("anomalythreshold", self.DEFAULTS["AnomalyThreshold"]))

        with open(self.model_path, "rb") as f:
            artifact = pickle.load(f)

        self.schema = artifact["schema"]
        self.include_deltas = artifact.get("config", {}).get("include_deltas", True)

        # Per-scenario layout takes precedence; legacy single-model is fallback.
        if "models" in artifact:
            models = artifact["models"]
            if self.scenario not in models:
                raise ValueError(
                    f"Scenario {self.scenario!r} not in pickle; "
                    f"available: {sorted(models)}"
                )
            self.model = models[self.scenario]
            print(f"[iforest] per-scenario layout, routing through {self.scenario!r} IF "
                  f"({len(models)} models in pickle)")
        elif "model" in artifact:
            self.model = artifact["model"]
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
        self._scalar_indices = [self._header_to_idx[c] for c in self.schema["scalar_columns"]
                                if c in self._header_to_idx]
        self._list_columns: list[tuple[int, list[list[int]]]] = []
        for col, layout in self.schema["list_columns"].items():
            if col not in self._header_to_idx:
                continue
            self._list_columns.append((self._header_to_idx[col], layout["paths"]))

        self.n_raw = len(self._scalar_indices) + sum(len(p) for _, p in self._list_columns)
        self.prev_raw: np.ndarray | None = None
        self._latest_score: float | None = None
        self._latest_is_anomaly: bool = False

        # Operational logging — until csv_output is rerouted through the
        # complex-reasoning tier, this is the only way to observe the
        # plugin's per-frame output online. Prints every N-th frame plus
        # every anomaly transition.
        self._frame_count = 0
        self._heartbeat_every = int(cfg.get("heartbeatevery", "100"))
        self._was_anomaly = False

        print(f"[iforest] threshold={self.threshold:+.4f} "
              f"(score < threshold ⇒ anomaly), n_raw_features={self.n_raw}, "
              f"heartbeat every {self._heartbeat_every} frames")

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

    def update(self, low_level_data=None, high_level_data=None):
        if not low_level_data:
            return
        raw = self._frame_to_raw(low_level_data)

        if self.include_deltas:
            if self.prev_raw is None:
                delta = np.zeros_like(raw)
            else:
                delta = raw - self.prev_raw
            features = np.concatenate([raw, delta])
        else:
            features = raw

        self.prev_raw = raw
        score = float(self.model.decision_function(features.reshape(1, -1))[0])
        is_anomaly = score < self.threshold
        self._latest_score = score
        self._latest_is_anomaly = is_anomaly
        self._frame_count += 1
        # Edge-triggered alert on transitions, plus a heartbeat so silence
        # is distinguishable from "plugin alive but everything nominal".
        if is_anomaly and not self._was_anomaly:
            print(f"[iforest][ALERT] frame={self._frame_count} "
                  f"scenario={self.scenario} score={score:+.4f} "
                  f"threshold={self.threshold:+.4f} (entered anomaly)")
        elif self._was_anomaly and not is_anomaly:
            print(f"[iforest][CLEAR] frame={self._frame_count} "
                  f"scenario={self.scenario} score={score:+.4f} "
                  f"(left anomaly)")
        elif self._heartbeat_every > 0 and self._frame_count % self._heartbeat_every == 0:
            print(f"[iforest] frame={self._frame_count} "
                  f"scenario={self.scenario} score={score:+.4f} "
                  f"is_anomaly={is_anomaly}")
        self._was_anomaly = is_anomaly

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
