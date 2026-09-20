# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
# Licensed under the NASA Open Source Agreement version 1.3

"""Tests for the XGBoost attack-classifier plugin.

The plugin's behavior is shaped by the IF-gating decision: if the matching
per-mode IF says nominal, the classifier should never fire; if anomaly,
the classifier should emit a top-K prediction. These tests stub both the
IF and the classifier so the gating logic is exercised in isolation."""

import csv
import glob
import json
import os
import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest

from plugins.xgb_classifier.xgb_classifier_plugin import Plugin as XgbPlugin


class _FakeIF:
    """Picklable IF stub with a steerable decision_function."""
    def __init__(self, score: float = 0.05):
        self.score = score
    def decision_function(self, X):
        return np.full(len(X), self.score, dtype=float)


class _FakeClassifier:
    """Picklable classifier stub returning fixed probabilities per row.

    Exposes `classes_` (like a real sklearn estimator) so the plugin labels
    predictions by the head's own class list — the same order as `probs`."""
    def __init__(self, probs: list[float], n_features: int = 4, classes=None):
        self.probs = np.asarray(probs, dtype=float)
        self.n_features = n_features
        self.classes_ = np.asarray(
            classes if classes is not None
            else [f"c{i}" for i in range(len(self.probs))])
    def predict_proba(self, X):
        return np.tile(self.probs, (len(X), 1))


def _build_artifacts(tmp_path, *, classifier_probs, if_score: float,
                     if_threshold: float, scalar_cols, labels,
                     delta_only_cols=None, fallback_score: float | None = None,
                     cal_top_k: int = 3, cal_min_confidence: float = 0.30,
                     mode_heads=None, route_modes=None, calibration=None):
    """Pickle a (classifier, if) pair + matching calibration JSONs.

    `mode_heads` (a {mode: (probs, classes)} dict), `route_modes`, and
    `calibration` add the AINOS3-37 hybrid keys; omitted → a plain v3 artifact."""
    # Classifier pickle
    cls_pkl = tmp_path / "cls.pkl"
    cls_art = {
        "clf": _FakeClassifier(classifier_probs, n_features=len(scalar_cols),
                               classes=labels),
        "labels": list(labels),
        "label_to_id": {l: i for i, l in enumerate(labels)},
        "schema": {
            "scalar_columns": list(scalar_cols),
            "list_columns": {},
            "delta_only_columns": list(delta_only_cols or []),
            "feature_names": list(scalar_cols),
        },
        "config": {"include_deltas": True},
    }
    if mode_heads is not None:
        cls_art["mode_heads"] = {
            m: _FakeClassifier(probs, n_features=len(scalar_cols), classes=cls)
            for m, (probs, cls) in mode_heads.items()}
        cls_art["route_modes"] = list(route_modes or mode_heads.keys())
        cls_art["calibration"] = calibration or {}
    with open(cls_pkl, "wb") as f:
        pickle.dump(cls_art, f)
    cls_cal = tmp_path / "cls.calibration.json"
    cls_cal.write_text(json.dumps(
        {"top_k": cal_top_k, "min_confidence": cal_min_confidence}))

    # IF pickle
    if_pkl = tmp_path / "if.pkl"
    fallback = fallback_score if fallback_score is not None else if_score
    if_art = {
        "models": {
            "MODE_PASSIVE": _FakeIF(score=if_score),
            "MODE_BDOT": _FakeIF(score=if_score),
            "MODE_SUNSAFE": _FakeIF(score=fallback),
            "MODE_INERTIAL": _FakeIF(score=if_score),
        },
        "schema": {
            "scalar_columns": list(scalar_cols),
            "list_columns": {},
            "delta_only_columns": list(delta_only_cols or []),
            "feature_names": list(scalar_cols),
        },
        "config": {"include_deltas": True},
    }
    with open(if_pkl, "wb") as f:
        pickle.dump(if_art, f)
    if_cal = tmp_path / "if.calibration.json"
    if_cal.write_text(json.dumps(
        {"thresholds": {"MODE_PASSIVE": if_threshold, "MODE_BDOT": if_threshold,
                        "MODE_SUNSAFE": if_threshold,
                        "MODE_INERTIAL": if_threshold},
         "target_fp_rate": 0.01}))
    return cls_pkl, cls_cal, if_pkl, if_cal


@pytest.fixture
def configured(tmp_path, monkeypatch):
    """Pickle stubs + write an ini pointing at them.

    Returns a build fn so individual tests can vary scores/thresholds/labels."""

    def _build(*, if_score: float = 0.05, if_threshold: float = 0.0,
               classifier_probs: list[float] | None = None,
               labels: list[str] | None = None,
               headers: list[str] | None = None,
               warmup_frames: int = 0, top_k: int = 3,
               min_confidence: float = 0.30, write_side: str = "true",
               flush_every: int = 1, mode_heads=None, route_modes=None,
               calibration=None):
        scalar_cols = ["foo", "bar"]
        # Routing source header lives outside the schema (it's a mode flag).
        if headers is None:
            headers = scalar_cols + ["ADCS_GNC.Mode"]
        if labels is None:
            labels = ["nominal", "EX-0001.01", "EX-0012.04"]
        if classifier_probs is None:
            classifier_probs = [0.10, 0.20, 0.70]
        cls_pkl, cls_cal, if_pkl, if_cal = _build_artifacts(
            tmp_path, classifier_probs=classifier_probs, if_score=if_score,
            if_threshold=if_threshold, scalar_cols=scalar_cols, labels=labels,
            cal_top_k=top_k, cal_min_confidence=min_confidence,
            mode_heads=mode_heads, route_modes=route_modes, calibration=calibration)
        ini = tmp_path / "test.ini"
        ini.write_text(
            "[XGB_CLASSIFIER]\n"
            f"ClassifierPath = {cls_pkl}\n"
            f"CalibrationPath = {cls_cal}\n"
            f"IfModelPath = {if_pkl}\n"
            f"IfCalibrationPath = {if_cal}\n"
            f"WriteSideFile = {write_side}\n"
            f"SideFileOutputDir = {tmp_path}\n"
            f"SideFileFlushEvery = {flush_every}\n"
            f"TopK = {top_k}\n"
            f"MinConfidence = {min_confidence}\n"
            f"WarmupFrames = {warmup_frames}\n"
            "HeartbeatEvery = 0\n"
            "RoutingSourceHeader = ADCS_GNC.Mode\n"
            'RoutingModeMap = {"0":"MODE_PASSIVE","1":"MODE_BDOT","2":"MODE_SUNSAFE","3":"MODE_INERTIAL"}\n'
            "FallbackMode = MODE_SUNSAFE\n"
        )
        monkeypatch.setenv("ONAIR_INI_FILE", str(ini))
        return XgbPlugin(MagicMock(), headers)

    return _build


def _read_side_file(tmp_path):
    files = glob.glob(os.path.join(str(tmp_path), "attack_class_*.csv"))
    assert len(files) <= 1, f"expected at most one side file, got {files}"
    if not files:
        return None, []
    with open(files[0]) as f:
        rows = list(csv.reader(f))
    return files[0], rows


def test_init_loads_both_pickles_and_writes_header(configured, tmp_path):
    plugin = configured()
    # Both models were loaded successfully + labels exposed.
    assert len(plugin.labels) == 3
    assert "MODE_SUNSAFE" in plugin.if_models
    # Side file not yet written (no frame processed)
    path, _ = _read_side_file(tmp_path)
    assert path is None


def test_if_below_threshold_skips_classification(configured, tmp_path):
    """IF says nominal (score >= threshold) → classifier must NOT fire.
    Side-file records the skip with if_anomaly=0 and empty class column."""
    plugin = configured(if_score=0.05, if_threshold=0.0,
                        classifier_probs=[0.10, 0.20, 0.70])
    plugin.update(low_level_data=["1.0", "2.0", "2"])  # mode=2 → SUNSAFE
    reasoning = plugin.render_reasoning()
    assert reasoning["is_anomaly"] is False
    assert reasoning["predicted_class"] is None
    assert reasoning["skipped_reason"] == "if_below_threshold"

    _, rows = _read_side_file(tmp_path)
    assert len(rows) == 2  # header + 1 skip row
    assert rows[0][:5] == ["frame_idx", "mode", "if_score", "if_anomaly",
                           "predicted_class"]
    assert rows[1][3] == "0"   # if_anomaly
    assert rows[1][4] == ""    # predicted_class empty


def test_if_above_threshold_runs_classifier_emits_top_k(configured, tmp_path):
    """IF flags anomaly → classifier runs, top-K populated in side-file."""
    plugin = configured(if_score=-0.05, if_threshold=0.0,
                        classifier_probs=[0.10, 0.20, 0.70])
    plugin.update(low_level_data=["1.0", "2.0", "2"])
    r = plugin.render_reasoning()
    assert r["is_anomaly"] is True
    assert r["predicted_class"] == "EX-0012.04"  # 0.70 wins
    # AINOS3-26: with no taxonomy loaded, the cluster maps to the class itself.
    assert r["predicted_cluster"] == "EX-0012.04"
    assert r["predicted_confidence"] == pytest.approx(0.70)
    assert [c for c, _ in r["predictions"]] == ["EX-0012.04", "EX-0001.01",
                                                 "nominal"]

    _, rows = _read_side_file(tmp_path)
    data = rows[1]
    # frame_idx=0, mode=MODE_SUNSAFE, if_anomaly=1, predicted=EX-0012.04,
    # predicted_cluster (AINOS3-26 column 5), then top-K pairs from column 6.
    assert data[3] == "1"
    assert data[4] == "EX-0012.04"     # predicted_class
    assert data[5] == "EX-0012.04"     # predicted_cluster (self-map, no taxonomy)
    assert data[6] == "EX-0012.04"     # top1_class
    assert float(data[7]) == pytest.approx(0.70)  # top1_prob


def test_below_min_confidence_emits_unknown(configured, tmp_path):
    """When the winning class probability is below min_confidence, the
    plugin marks the prediction as `unknown` but still records the top-K
    in the side-file so analysts can audit borderline cases."""
    plugin = configured(if_score=-0.05, if_threshold=0.0,
                        classifier_probs=[0.34, 0.33, 0.33],  # max=0.34
                        min_confidence=0.50)
    plugin.update(low_level_data=["1.0", "2.0", "2"])
    r = plugin.render_reasoning()
    assert r["predicted_class"] == "unknown"
    assert r["predicted_confidence"] == pytest.approx(0.34)

    _, rows = _read_side_file(tmp_path)
    data = rows[1]
    assert data[4] == "unknown"
    # but the actual top-1 class still surfaces in the top-K columns
    assert data[5] == "nominal"  # alphabetically nominal wins tie at 0.34


def test_warmup_suppresses_all_classifications(configured, tmp_path):
    """During warmup, never emit a classification — let prev_raw settle.
    Side-file stays empty during warmup."""
    plugin = configured(if_score=-0.05, if_threshold=0.0,
                        classifier_probs=[0.10, 0.20, 0.70],
                        warmup_frames=3)
    for _ in range(3):
        plugin.update(low_level_data=["1.0", "2.0", "2"])
        r = plugin.render_reasoning()
        assert r.get("skipped_reason") == "warmup"
        assert r["predicted_class"] is None

    # 4th frame: post-warmup, classifier fires
    plugin.update(low_level_data=["1.0", "2.0", "2"])
    r = plugin.render_reasoning()
    assert r["is_anomaly"] is True
    assert r["predicted_class"] == "EX-0012.04"

    _, rows = _read_side_file(tmp_path)
    # Header + 1 data row (only the post-warmup frame).
    assert len(rows) == 2


def test_routing_per_mode_uses_correct_if(configured, tmp_path):
    """Each ADCS mode resolves to its own per-mode IF. Switching the mode
    header value between frames routes through different IF models."""
    # All IFs share score=-0.05 with threshold=0.0 → all anomalies. Verify
    # the mode column in the side-file reflects the routed mode.
    plugin = configured(if_score=-0.05, if_threshold=0.0)
    plugin.update(low_level_data=["1.0", "2.0", "0"])  # PASSIVE
    plugin.update(low_level_data=["1.0", "2.0", "1"])  # BDOT
    plugin.update(low_level_data=["1.0", "2.0", "2"])  # SUNSAFE
    plugin.update(low_level_data=["1.0", "2.0", "3"])  # INERTIAL

    _, rows = _read_side_file(tmp_path)
    modes = [r[1] for r in rows[1:]]
    assert modes == ["MODE_PASSIVE", "MODE_BDOT", "MODE_SUNSAFE",
                     "MODE_INERTIAL"]


def test_unmapped_mode_falls_back_to_fallback(configured, tmp_path):
    """An ADCS_GNC.Mode value not in RoutingModeMap routes through
    FallbackMode (defaults to MODE_SUNSAFE)."""
    plugin = configured(if_score=-0.05, if_threshold=0.0)
    plugin.update(low_level_data=["1.0", "2.0", "99"])  # not in map
    _, rows = _read_side_file(tmp_path)
    assert rows[1][1] == "MODE_SUNSAFE"


def test_routing_init_rejects_map_targeting_missing_mode(tmp_path, monkeypatch):
    """Misconfigured RoutingModeMap (target not in IF pickle) is a deploy
    error and must raise at init time, not silently fail at scoring."""
    scalar_cols = ["foo", "bar"]
    labels = ["nominal", "EX-0001.01"]
    cls_pkl, cls_cal, if_pkl, if_cal = _build_artifacts(
        tmp_path, classifier_probs=[0.5, 0.5], if_score=0.0,
        if_threshold=0.0, scalar_cols=scalar_cols, labels=labels)
    ini = tmp_path / "test.ini"
    ini.write_text(
        "[XGB_CLASSIFIER]\n"
        f"ClassifierPath = {cls_pkl}\n"
        f"CalibrationPath = {cls_cal}\n"
        f"IfModelPath = {if_pkl}\n"
        f"IfCalibrationPath = {if_cal}\n"
        "WriteSideFile = false\n"
        f"SideFileOutputDir = {tmp_path}\n"
        'RoutingModeMap = {"7":"MODE_GHOST"}\n'
    )
    monkeypatch.setenv("ONAIR_INI_FILE", str(ini))
    with pytest.raises(ValueError, match="RoutingModeMap targets modes not in"):
        XgbPlugin(MagicMock(), scalar_cols + ["ADCS_GNC.Mode"])


def test_disable_side_file_suppresses_output(configured, tmp_path):
    plugin = configured(if_score=-0.05, if_threshold=0.0,
                        write_side="false")
    assert plugin._side_file_path is None
    plugin.update(low_level_data=["1.0", "2.0", "2"])
    files = glob.glob(os.path.join(str(tmp_path), "attack_class_*.csv"))
    assert files == []


def test_empty_frame_is_noop(configured, tmp_path):
    """Empty low_level_data short-circuits without advancing the frame
    counter — keeps side-file frame_idx aligned with csv_output."""
    plugin = configured(if_score=-0.05, if_threshold=0.0)
    plugin.update(low_level_data=[])
    plugin.update(low_level_data=None)
    assert plugin._frame_count == 0
    path, _ = _read_side_file(tmp_path)
    assert path is None


def test_render_reasoning_before_first_frame_returns_safe_default(configured):
    """render_reasoning called before any update() must not raise — it's
    invoked by OnAIR's vehicle_rep early in the lifecycle."""
    plugin = configured(if_score=-0.05, if_threshold=0.0)
    r = plugin.render_reasoning()
    assert r["is_anomaly"] is False
    assert r["predicted_class"] is None
    assert r["predictions"] == []


# ─── AINOS3-37: selective per-mode hybrid ────────────────────────────────
# labels: global head favors EX-0012.04; the MODE_INERTIAL per-mode head
# carries only a subset {nominal, EX-0001.01} and favors EX-0001.01.
_HY_LABELS = ["nominal", "EX-0001.01", "EX-0012.04"]
_HY_MODE_HEADS = {
    # (probs, classes) — a real per-mode head sees only its mode's classes.
    "MODE_INERTIAL": ([0.2, 0.8], ["nominal", "EX-0001.01"]),
}


def test_hybrid_backward_compat_plain_artifact_has_no_routing(configured):
    """A plain v3 artifact (no hybrid keys) loads with empty routing — the
    single global head serves every mode, unchanged behaviour."""
    plugin = configured(if_score=-0.05, if_threshold=0.0)
    assert plugin.mode_heads == {}
    assert plugin.route_modes == set()
    assert plugin.calibration == {}


def test_hybrid_routes_dynamic_mode_to_per_mode_head(configured):
    """A routed mode (INERTIAL, mode=3) is scored by its per-mode head, whose
    top class (EX-0001.01) differs from the global head's (EX-0012.04)."""
    plugin = configured(
        if_score=-0.05, if_threshold=0.0,
        classifier_probs=[0.10, 0.20, 0.70], labels=_HY_LABELS,
        mode_heads=_HY_MODE_HEADS, route_modes=["MODE_INERTIAL"])
    plugin.update(low_level_data=["1.0", "2.0", "3"])   # mode=3 → INERTIAL
    r = plugin.render_reasoning()
    assert r["mode"] == "MODE_INERTIAL"
    assert r["predicted_class"] == "EX-0001.01"          # from the per-mode head
    assert r["predictions"][0][0] == "EX-0001.01"


def test_hybrid_nonrouted_mode_uses_global_head(configured):
    """A non-routed mode (SUNSAFE, mode=2) keeps the global head — baseline
    behaviour, so its top class is the global head's EX-0012.04."""
    plugin = configured(
        if_score=-0.05, if_threshold=0.0,
        classifier_probs=[0.10, 0.20, 0.70], labels=_HY_LABELS,
        mode_heads=_HY_MODE_HEADS, route_modes=["MODE_INERTIAL"])
    plugin.update(low_level_data=["1.0", "2.0", "2"])   # mode=2 → SUNSAFE
    r = plugin.render_reasoning()
    assert r["mode"] == "MODE_SUNSAFE"
    assert r["predicted_class"] == "EX-0012.04"          # global head


def test_hybrid_per_mode_calibration_scales_confidence(configured):
    """Per-mode calibration is a monotone np.interp map on the top-1 prob: it
    changes the reported confidence but never the winning class. Map
    (0→0, 1→0.5) sends the per-mode head's raw 0.8 → 0.4."""
    plugin = configured(
        if_score=-0.05, if_threshold=0.0,
        classifier_probs=[0.10, 0.20, 0.70], labels=_HY_LABELS,
        mode_heads=_HY_MODE_HEADS, route_modes=["MODE_INERTIAL"],
        calibration={"MODE_INERTIAL": {"x": [0.0, 1.0], "y": [0.0, 0.5]}},
        min_confidence=0.30)
    plugin.update(low_level_data=["1.0", "2.0", "3"])   # INERTIAL, raw top1=0.8
    r = plugin.render_reasoning()
    assert r["predicted_class"] == "EX-0001.01"          # argmax unchanged
    assert r["predicted_confidence"] == pytest.approx(0.40)  # 0.8 → calibrated 0.4


def test_hybrid_calibration_below_min_confidence_emits_unknown(configured):
    """If calibration pushes the top-1 confidence under min_confidence, the
    class is reported 'unknown' (the gate uses the calibrated number)."""
    plugin = configured(
        if_score=-0.05, if_threshold=0.0,
        classifier_probs=[0.10, 0.20, 0.70], labels=_HY_LABELS,
        mode_heads=_HY_MODE_HEADS, route_modes=["MODE_INERTIAL"],
        calibration={"MODE_INERTIAL": {"x": [0.0, 1.0], "y": [0.0, 0.25]}},
        min_confidence=0.30)
    plugin.update(low_level_data=["1.0", "2.0", "3"])   # raw 0.8 → calibrated 0.2
    r = plugin.render_reasoning()
    assert r["predicted_confidence"] == pytest.approx(0.20)
    assert r["predicted_class"] == "unknown"
