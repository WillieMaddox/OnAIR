# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
# Licensed under the NASA Open Source Agreement version 1.3
"""Isolation Forest Learner plugin (Tier 1 — sketch, not yet registered).

Loads a pickled IsolationForest + feature schema produced by
`components/onair/training/train.py` and scores each incoming frame.

To activate:
1. Train: `python3 components/onair/training/train.py`
2. Add to `nos3_security.ini`:
       LearnersPluginDict = {'iforest': 'cf/onair/plugins/isolation_forest/__init__.py'}
3. Add the model path under [ISOLATION_FOREST]:
       ModelPath = data/onair/models/iforest_v1.pkl
4. Re-run the OnAIR build/sync (CMakeLists copies plugins to fsw/build/exe/cpu1/cf/onair/plugins/)

Returns `{"anomaly_score": float, "is_anomaly": bool}` from render_reasoning().
The csv_output plugin will append these as additional columns when the learner
is in `LearnersPluginDict`.
"""

from __future__ import annotations

import ast
import configparser
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
        "ModelPath": "data/onair/models/iforest_v1.pkl",
        "AnomalyThreshold": "0.0",  # decision_function < threshold → anomaly
    }

    def __init__(self, name, headers):
        super().__init__(name, headers)

        cfg = self._load_config()
        self.model_path = cfg.get("modelpath", self.DEFAULTS["ModelPath"])
        self.threshold = float(cfg.get("anomalythreshold", self.DEFAULTS["AnomalyThreshold"]))

        with open(self.model_path, "rb") as f:
            artifact = pickle.load(f)
        self.model = artifact["model"]
        self.schema = artifact["schema"]
        self.include_deltas = artifact["config"]["include_deltas"]

        # Build header index for fast frame -> feature mapping.
        self._header_to_idx = {h: i for i, h in enumerate(headers)}
        self._scalar_indices = [self._header_to_idx[c] for c in self.schema["scalar_columns"]
                                if c in self._header_to_idx]
        # List columns: each entry is (frame_idx, paths) so we can extract leaves
        # in the canonical order recorded at training time.
        self._list_columns: list[tuple[int, list[list[int]]]] = []
        for col, layout in self.schema["list_columns"].items():
            if col not in self._header_to_idx:
                continue
            self._list_columns.append((self._header_to_idx[col], layout["paths"]))

        self.n_raw = len(self._scalar_indices) + sum(len(p) for _, p in self._list_columns)
        self.prev_raw: np.ndarray | None = None
        self._latest_score: float | None = None
        self._latest_is_anomaly: bool = False

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
        self._latest_score = score
        self._latest_is_anomaly = score < self.threshold

    def render_reasoning(self):
        if self._latest_score is None:
            return {"anomaly_score": 0.0, "is_anomaly": False}
        return {
            "anomaly_score": self._latest_score,
            "is_anomaly": self._latest_is_anomaly,
        }
