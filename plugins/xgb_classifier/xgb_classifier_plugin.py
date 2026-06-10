# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
# Licensed under the NASA Open Source Agreement version 1.3
"""Attack-classifier Learner plugin (Tier 1, Phase 3).

Sits DOWNSTREAM of the per-ADCS-mode Isolation Forest. On every frame:

  1. Build the same 894-feature vector the IF sees (raw + deltas, with
     delta_only_columns masked out of the raw half).
  2. Score that vector through the matching per-mode IsolationForest from
     the v5 pickle. If the score is at or above the calibrated per-mode
     threshold (i.e. the IF says "nominal"), skip classification entirely
     — the operational cost is then one IF score call per frame.
  3. If the IF score falls below the threshold (anomaly), run the
     XGBoost-equivalent (sklearn HistGradientBoostingClassifier) and emit
     top-K class predictions in `render_reasoning()` + the side-file.

The side-file lives next to the IF's `iforest_out_*.csv` and the CSV
writer's `csv_out_*.csv`. Filename: `attack_class_<ts>_pid<N>.csv`.
Columns: frame_idx, scenario, if_score, if_anomaly, top1_class, top1_prob,
top2_class, top2_prob, top3_class, top3_prob.

This plugin loads BOTH the v5 IF pickle (for gating) AND its own XGBoost
pickle. Loading two pickles is intentional — keeping the gating IF inside
the classifier plugin means it doesn't depend on cross-plugin data flow
through OnAIR's high_level_data path.

Activation:
1. Train: see `components/onair/training/train_classifier.py` (training
   pipeline that produced `xgb_attack_classifier_v1.pkl`).
2. `nos3_security.ini`:
       LearnersPluginDict = {
         'isolation_forest': 'cf/onair/plugins/isolation_forest/__init__.py',
         'xgb_classifier':   'cf/onair/plugins/xgb_classifier/__init__.py',
       }

       [XGB_CLASSIFIER]
       ClassifierPath = data/onair/models/xgb_attack_classifier_v1.pkl
       CalibrationPath =        # empty ⇒ derive as <model>.calibration.json
       IfModelPath = data/onair/models/iforest_per_mode_v5_invariant_bolstered.pkl
       IfCalibrationPath =
       # IF-gating mode resolution copied from the IF plugin's RuntimeRouting
       # so this plugin and the IF stay in lock-step on which model scored
       # which row.
       RoutingSourceHeader = ADCS_GNC.Mode
       RoutingModeMap = {"0":"MODE_PASSIVE","1":"MODE_BDOT","2":"MODE_SUNSAFE","3":"MODE_INERTIAL"}
3. Sync: re-run the OnAIR build (CMakeLists copies plugins to
   fsw/build/exe/cpu1/cf/onair/plugins/) plus the pickles + JSONs into
   a path accessible at runtime cwd.
"""

from __future__ import annotations

import ast
import configparser
import csv
import json
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np

# The incident aggregator (NOS3-201) ships in this plugin package; make the
# package dir importable regardless of how OnAIR loads the plugin file.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from incident import IncidentAggregator, cluster_map_from_taxonomy  # noqa: E402

from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin


def _coerce_scalar(value: Any) -> float:
    if value in ("", "[0]", "nan", None):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_list_value(value: Any):
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
    """Attack classifier scorer with IF gating.

    Reuses the IF plugin's feature-extraction recipe (raw + deltas,
    delta_only_columns masked) so the XGBoost model sees exactly the
    same 894-feature vector at runtime as it did at training time.
    """

    DEFAULTS = {
        "ClassifierPath": "data/onair/models/xgb_attack_classifier_v1.pkl",
        "CalibrationPath": "",  # empty → derive as <model>.calibration.json
        "IfModelPath": "data/onair/models/iforest_per_mode_v5_invariant_bolstered.pkl",
        "IfCalibrationPath": "",
        "WriteSideFile": "true",
        "SideFileOutputDir": "../../../../data/onair/csv",
        "SideFileFlushEvery": "10",
        "TopK": "3",
        "MinConfidence": "0.30",
        "RoutingSourceHeader": "ADCS_GNC.Mode",
        # JSON-encoded {mode_value_as_string: scenario_name}
        "RoutingModeMap": '{"0":"MODE_PASSIVE","1":"MODE_BDOT","2":"MODE_SUNSAFE","3":"MODE_INERTIAL"}',
        # Cold-start fallback mode (matches IF plugin default).
        "FallbackMode": "MODE_SUNSAFE",
        # Heartbeat / startup-transient suppression mirrors the IF plugin
        # so this plugin doesn't emit spurious classifications during the
        # first N frames before SBN subscriptions are fully populated.
        "WarmupFrames": "30",
        "HeartbeatEvery": "100",
        # ─── NOS3-202: cluster reporting ─────────────────────────────
        # Telemetry-indistinguishable sub-techniques are reported as a single
        # cluster (e.g. "EX-0012.{03,04,05}"). Empty ⇒ derive from the
        # classifier path's sibling cluster_rescore dir; missing file ⇒ each
        # class is its own cluster (no behavioural change).
        "ClusterTaxonomyPath": "",
        # ─── NOS3-201: incident aggregation ──────────────────────────
        "WriteIncidentFile": "true",
        # Hysteresis for incident open/close. Mirrors the IF plugin's
        # alert/clear hysteresis so an incident == one operational alert.
        "AlertHysteresis": "3",
        "ClearHysteresis": "5",
        "IncidentMinAnomalyFrames": "1",
    }

    def __init__(self, name, headers):
        super().__init__(name, headers)
        cfg = self._load_config()

        # ─── Load XGBoost classifier ──────────────────────────────────
        self.classifier_path = cfg.get("classifierpath", self.DEFAULTS["ClassifierPath"])
        cal_path = cfg.get("calibrationpath", self.DEFAULTS["CalibrationPath"]).strip()
        if not cal_path:
            cal_path = self.classifier_path.removesuffix(".pkl") + ".calibration.json"
        self.classifier_cal_path = cal_path

        with open(self.classifier_path, "rb") as f:
            cls_art = pickle.load(f)
        self.clf = cls_art["clf"]
        self.labels: list[str] = list(cls_art["labels"])
        self.schema = cls_art["schema"]
        self.include_deltas = cls_art.get("config", {}).get("include_deltas", True)

        # Load classifier calibration (top_k, min_confidence)
        self.top_k = int(cfg.get("topk", self.DEFAULTS["TopK"]))
        self.min_confidence = float(cfg.get("minconfidence", self.DEFAULTS["MinConfidence"]))
        cal_top_k, cal_min_conf = self._read_classifier_cal(self.classifier_cal_path)
        if cal_top_k is not None:
            self.top_k = cal_top_k
        if cal_min_conf is not None:
            self.min_confidence = cal_min_conf

        # ─── NOS3-202: cluster taxonomy ───────────────────────────────
        # Map each sub-technique to its telemetry-indistinguishable cluster.
        # Unknown / unclustered classes map to themselves.
        tax_path = cfg.get("clustertaxonomypath",
                           self.DEFAULTS["ClusterTaxonomyPath"]).strip()
        if not tax_path:
            base = os.path.dirname(self.classifier_path)
            tax_path = os.path.join(base, "cluster_rescore", "cluster_taxonomy.json")
        self._cluster_map: dict[str, str] = {}
        if os.path.exists(tax_path):
            try:
                with open(tax_path) as f:
                    self._cluster_map = cluster_map_from_taxonomy(json.load(f))
                print(f"[xgb_cls] cluster taxonomy: {os.path.basename(tax_path)} "
                      f"({len(self._cluster_map)} mapped classes)")
            except (OSError, json.JSONDecodeError, KeyError) as e:
                print(f"[xgb_cls] WARNING: cluster taxonomy unreadable ({e!r}); "
                      f"reporting raw sub-techniques")
        else:
            print(f"[xgb_cls] no cluster taxonomy at {tax_path}; "
                  f"reporting raw sub-techniques")

        # ─── Load gating IF + per-mode thresholds ────────────────────
        self.if_path = cfg.get("ifmodelpath", self.DEFAULTS["IfModelPath"])
        if_cal_path = cfg.get("ifcalibrationpath", self.DEFAULTS["IfCalibrationPath"]).strip()
        if not if_cal_path:
            if_cal_path = self.if_path.removesuffix(".pkl") + ".calibration.json"
        self.if_cal_path = if_cal_path
        with open(self.if_path, "rb") as f:
            if_art = pickle.load(f)
        self.if_models: dict[str, Any] = dict(if_art.get("models") or {})
        self.if_thresholds = self._load_if_thresholds(if_cal_path)
        # Cold-start fallback: if a row arrives before ADCS_GNC.Mode is
        # populated, gate using the fallback mode's threshold so we don't
        # silently skip every pre-mode frame.
        self.fallback_mode = cfg.get("fallbackmode", self.DEFAULTS["FallbackMode"])

        # ─── Build header index for feature extraction ────────────────
        # Mirrors isolation_forest_plugin._frame_to_raw exactly so the
        # XGBoost classifier sees the SAME feature vector the v5 IF saw
        # at training time — bit-equal layout.
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
        if delta_only_set:
            mask = np.zeros(self.n_raw, dtype=bool)
            for i, is_do in enumerate(scalar_delta_only_flags):
                mask[i] = is_do
            offset = len(self._scalar_indices)
            for (_, paths), is_do in zip(self._list_columns, list_delta_only_flags):
                if is_do:
                    mask[offset:offset + len(paths)] = True
                offset += len(paths)
            self._raw_keep_mask = ~mask
        else:
            self._raw_keep_mask = None
        self.prev_raw: np.ndarray | None = None

        # ─── Mode-routing config ─────────────────────────────────────
        self._routing_source = cfg.get(
            "routingsourceheader", self.DEFAULTS["RoutingSourceHeader"]).strip()
        try:
            raw_map = cfg.get("routingmodemap", self.DEFAULTS["RoutingModeMap"])
            self._routing_map: dict[str, str] = {
                str(k): str(v) for k, v in json.loads(raw_map).items()
            }
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"[xgb_cls][route] WARNING: RoutingModeMap parse failed ({e!r}); "
                  f"falling back to {self.fallback_mode!r} for every frame")
            self._routing_map = {}
        self._routing_source_idx: int | None = (
            self._header_to_idx.get(self._routing_source))

        # Bad config is loud at startup.
        unknown_modes = [v for v in self._routing_map.values() if v not in self.if_models]
        if unknown_modes:
            raise ValueError(
                f"RoutingModeMap targets modes not in IF pickle: "
                f"{sorted(set(unknown_modes))}; "
                f"available: {sorted(self.if_models)}")
        if self.fallback_mode not in self.if_models:
            raise ValueError(
                f"FallbackMode {self.fallback_mode!r} not in IF pickle; "
                f"available: {sorted(self.if_models)}")

        # ─── Operational state ───────────────────────────────────────
        self._warmup_frames = int(cfg.get("warmupframes", self.DEFAULTS["WarmupFrames"]))
        self._heartbeat_every = int(cfg.get("heartbeatevery", self.DEFAULTS["HeartbeatEvery"]))
        self._frame_count = 0
        self._latest_reasoning: dict | None = None

        # ─── Side-file writer setup ──────────────────────────────────
        self._side_file_path: str | None = None
        self._side_file_buffer: list[list[Any]] = []
        self._side_file_header_written = False
        self._side_file_flush_every = int(
            cfg.get("sidefileflushevery", self.DEFAULTS["SideFileFlushEvery"]))
        write_side = cfg.get(
            "writesidefile", self.DEFAULTS["WriteSideFile"]).strip().lower() == "true"
        if write_side:
            side_dir = cfg.get(
                "sidefileoutputdir", self.DEFAULTS["SideFileOutputDir"])
            os.makedirs(side_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
            self._side_file_path = os.path.join(
                side_dir, f"attack_class_{ts}_pid{os.getpid()}.csv")

        # ─── NOS3-201: incident aggregator + incident side-file ───────
        self._incident_agg = IncidentAggregator(
            alert_hysteresis=int(cfg.get("alerthysteresis",
                                         self.DEFAULTS["AlertHysteresis"])),
            clear_hysteresis=int(cfg.get("clearhysteresis",
                                         self.DEFAULTS["ClearHysteresis"])),
            min_anomaly_frames=int(cfg.get("incidentminanomalyframes",
                                           self.DEFAULTS["IncidentMinAnomalyFrames"])),
        )
        self._incident_file_path: str | None = None
        self._incident_header_written = False
        self._n_incidents = 0
        write_inc = cfg.get(
            "writeincidentfile", self.DEFAULTS["WriteIncidentFile"]
        ).strip().lower() == "true"
        if write_inc and self._side_file_path is not None:
            side_dir = cfg.get("sidefileoutputdir", self.DEFAULTS["SideFileOutputDir"])
            ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
            self._incident_file_path = os.path.join(
                side_dir, f"incident_{ts}_pid{os.getpid()}.csv")

        print(f"[xgb_cls] classifier loaded: {len(self.labels)} classes, "
              f"top_k={self.top_k}, min_confidence={self.min_confidence:.2f}")
        print(f"[xgb_cls] IF-gating against {os.path.basename(self.if_path)} "
              f"({len(self.if_models)} per-mode IFs)")
        print(f"[xgb_cls] feature schema: n_raw={self.n_raw}, "
              f"raw_kept={self.n_raw if self._raw_keep_mask is None else int(self._raw_keep_mask.sum())}")
        if self._side_file_path is not None:
            print(f"[xgb_cls] side-file → {self._side_file_path} "
                  f"(flush every {self._side_file_flush_every} rows)")
        else:
            print("[xgb_cls] side-file writer disabled")
        if self._incident_file_path is not None:
            print(f"[xgb_cls] incident-file → {self._incident_file_path} "
                  f"(alert/clear hysteresis "
                  f"{self._incident_agg.alert_hyst}/{self._incident_agg.clear_hyst})")

    @staticmethod
    def _load_config() -> dict:
        ini_path = os.environ.get("ONAIR_INI_FILE") or "cf/onair/nos3_security.ini"
        if not os.path.exists(ini_path):
            return {}
        parser = configparser.ConfigParser()
        parser.read(ini_path)
        if parser.has_section("XGB_CLASSIFIER"):
            return dict(parser.items("XGB_CLASSIFIER"))
        return {}

    def _load_if_thresholds(self, cal_path: str) -> dict[str, float]:
        if not os.path.exists(cal_path):
            print(f"[xgb_cls] WARNING: IF calibration not found at {cal_path}; "
                  f"every frame will be treated as anomaly (no gating)")
            return {}
        try:
            with open(cal_path) as f:
                cal = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[xgb_cls] WARNING: could not read IF calibration {cal_path}: {e}")
            return {}
        thr = cal.get("thresholds") or {}
        return {str(k): float(v) for k, v in thr.items()}

    @staticmethod
    def _read_classifier_cal(path: str) -> tuple[int | None, float | None]:
        if not os.path.exists(path):
            return None, None
        try:
            with open(path) as f:
                cal = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None, None
        return cal.get("top_k"), cal.get("min_confidence")

    def _frame_to_raw(self, frame) -> np.ndarray:
        raw = np.zeros(self.n_raw, dtype=np.float64)
        for out_i, frame_i in enumerate(self._scalar_indices):
            raw[out_i] = _coerce_scalar(frame[frame_i])
        offset = len(self._scalar_indices)
        for frame_i, paths in self._list_columns:
            parsed = _parse_list_value(frame[frame_i])
            if parsed is not None:
                for j, path in enumerate(paths):
                    raw[offset + j] = _extract_at_path(parsed, path)
            offset += len(paths)
        return raw

    def _resolve_mode(self, frame) -> str:
        """Look up the current ADCS mode from the frame; fall back to
        FallbackMode when the source header is missing, the value is the
        SBN placeholder `[0]`, or the value isn't in RoutingModeMap."""
        if self._routing_source_idx is None:
            return self.fallback_mode
        raw_value = frame[self._routing_source_idx]
        key = str(raw_value).strip()
        return self._routing_map.get(key, self.fallback_mode)

    def _gate_via_if(self, features: np.ndarray, mode: str) -> tuple[bool, float]:
        """Return (is_anomaly, score) using the per-mode IF model for the
        active ADCS mode. If the mode isn't in the pickle (shouldn't
        happen — startup validation rejects this), defaults to anomaly=True
        so the classifier still gets a chance."""
        if mode not in self.if_models:
            return True, float("nan")
        score = float(self.if_models[mode].decision_function(features.reshape(1, -1))[0])
        threshold = self.if_thresholds.get(mode, 0.0)
        return score < threshold, score

    def update(self, low_level_data=None, high_level_data=None):
        if not low_level_data:
            return
        self._frame_count += 1
        raw = self._frame_to_raw(low_level_data)

        # Compute features in the IF's exact layout. include_deltas=True
        # for the v5 model; if the trained classifier had include_deltas
        # set to False we still need to honor that flag.
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

        # During cold-start warmup, never emit classifications. prev_raw is
        # established but the side-file stays clean.
        if self._frame_count <= self._warmup_frames:
            self._latest_reasoning = {
                "is_anomaly": False, "if_score": float("nan"),
                "mode": "warmup", "predicted_class": None,
                "predictions": [], "skipped_reason": "warmup",
            }
            return

        mode = self._resolve_mode(low_level_data)
        is_anomaly, if_score = self._gate_via_if(features, mode)

        frame_idx = self._frame_count - 1

        if not is_anomaly:
            # IF says nominal → skip classification entirely. Operational
            # cost = one IF score call per frame.
            self._latest_reasoning = {
                "is_anomaly": False, "if_score": if_score, "mode": mode,
                "predicted_class": None, "predicted_cluster": None,
                "predictions": [], "skipped_reason": "if_below_threshold",
            }
            # Feed the incident aggregator a nominal frame so its clear-
            # hysteresis advances (this is what eventually closes incidents).
            closed = self._incident_agg.update(frame_idx, is_anomaly=False, mode=mode)
            if closed is not None:
                self._handle_incident(closed)
            if self._side_file_path is not None:
                # Emit a row recording the skip so offline tools see the
                # cadence; predicted_class is empty, probs are 0.
                self._side_file_append_skip(if_score, mode)
            return

        # IF flagged → run classifier
        probs = self.clf.predict_proba(features.reshape(1, -1))[0]
        order = np.argsort(probs)[::-1]
        top = [(self.labels[i], float(probs[i])) for i in order[:self.top_k]]
        top1_class, top1_prob = top[0]
        # NOS3-202: collapse telemetry-indistinguishable sub-techniques.
        predicted_cluster = self._cluster_map.get(top1_class, top1_class)
        # Below-confidence override: emit "unknown" but keep the top probs
        # for post-hoc analysis.
        predicted_class = top1_class if top1_prob >= self.min_confidence else "unknown"

        self._latest_reasoning = {
            "is_anomaly": True, "if_score": if_score, "mode": mode,
            "predicted_class": predicted_class,
            "predicted_cluster": predicted_cluster,
            "predicted_confidence": top1_prob,
            "predictions": top,
        }

        # NOS3-201: feed the anomalous frame's cluster vote into the incident.
        closed = self._incident_agg.update(
            frame_idx, is_anomaly=True, mode=mode,
            cluster=predicted_cluster, sub_technique=top1_class,
            confidence=top1_prob)
        if closed is not None:
            self._handle_incident(closed)

        if self._side_file_path is not None:
            self._side_file_append(if_score, mode, predicted_class,
                                   predicted_cluster, top)
        if self._heartbeat_every > 0 and self._frame_count % self._heartbeat_every == 0:
            print(f"[xgb_cls] frame={self._frame_count} mode={mode} "
                  f"if_score={if_score:+.4f} → {predicted_cluster} "
                  f"({top1_prob*100:.0f}%)")

    def _handle_incident(self, inc) -> None:
        """Write a closed incident to the incident side-file + log a summary."""
        self._n_incidents += 1
        print(f"[xgb_cls][INCIDENT #{self._n_incidents}] "
              f"frames {inc.frame_start}-{inc.frame_end} "
              f"({inc.n_frames}f, {inc.n_anomaly_frames} anom) "
              f"mode={inc.mode} → {inc.cluster} "
              f"(conf={inc.confidence:.2f}, agree={inc.agreement:.0%})")
        if self._incident_file_path is None:
            return
        write_header = not self._incident_header_written
        with open(self._incident_file_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(inc.header())
                self._incident_header_written = True
            w.writerow(inc.as_row())

    def _side_file_append(self, if_score: float, mode: str,
                          predicted_class: str, predicted_cluster: str,
                          top: list[tuple[str, float]]):
        row: list[Any] = [
            self._frame_count - 1, mode, f"{if_score:.6f}", 1,
            predicted_class, predicted_cluster,
        ]
        # Flatten top-K into pairs (class, prob)
        for i in range(self.top_k):
            if i < len(top):
                row.extend([top[i][0], f"{top[i][1]:.6f}"])
            else:
                row.extend(["", ""])
        self._side_file_buffer.append(row)
        if len(self._side_file_buffer) >= self._side_file_flush_every:
            self._flush_side_file()

    def _side_file_append_skip(self, if_score: float, mode: str):
        # Skip rows record the cadence — IF-gating decisions are visible
        # offline. predicted_class/cluster empty, top_K columns empty.
        row: list[Any] = [
            self._frame_count - 1, mode, f"{if_score:.6f}", 0, "", "",
        ]
        for _ in range(self.top_k):
            row.extend(["", ""])
        self._side_file_buffer.append(row)
        if len(self._side_file_buffer) >= self._side_file_flush_every:
            self._flush_side_file()

    def _flush_side_file(self):
        if self._side_file_path is None or not self._side_file_buffer:
            return
        write_header = not self._side_file_header_written
        with open(self._side_file_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                header = ["frame_idx", "mode", "if_score", "if_anomaly",
                          "predicted_class", "predicted_cluster"]
                for i in range(self.top_k):
                    header.extend([f"top{i+1}_class", f"top{i+1}_prob"])
                w.writerow(header)
                self._side_file_header_written = True
            w.writerows(self._side_file_buffer)
        self._side_file_buffer.clear()

    def render_reasoning(self):
        if self._latest_reasoning is None:
            return {"is_anomaly": False, "predicted_class": None,
                    "predicted_cluster": None, "predictions": []}
        return dict(self._latest_reasoning)
