# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
# Licensed under the NASA Open Source Agreement version 1.3

"""Tests for the IsolationForest plugin's side-file writer.

Model loading and feature extraction are covered indirectly: each test
constructs the plugin against a tiny pickled artifact whose model is a
deterministic stub. The focus here is the persistence path — header line,
flush cadence, alert/cleared bits, and the disable switch."""

import csv
import glob
import json
import os
import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest

from plugins.isolation_forest.isolation_forest_plugin import Plugin as IF_Plugin


class _FakeIF:
    """Picklable IF stub with a steerable decision_function."""

    def __init__(self, score: float = 0.05):
        self.score = score

    def decision_function(self, X):
        return np.full(len(X), self.score, dtype=float)


def _make_artifact(score: float, scenario: str = "nominal_ops"):
    return {
        "models": {scenario: _FakeIF(score=score)},
        "schema": {
            "scalar_columns": ["foo", "bar"],
            "list_columns": {},
        },
        "config": {"include_deltas": False},
    }


@pytest.fixture
def configured(tmp_path, monkeypatch):
    """Pickle a fake artifact + calibration and point an ini at them.

    Returns a builder fn so individual tests can vary score/threshold/flush
    without re-pickling boilerplate."""

    def _build(*, score: float = 0.05, threshold: float = 0.0,
               flush_every: int = 2, write_side: str = "true",
               heartbeat: int = 0):
        pkl = tmp_path / "m.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(_make_artifact(score=score), f)
        cal = tmp_path / "m.calibration.json"
        cal.write_text(json.dumps({"thresholds": {"nominal_ops": threshold}}))
        ini = tmp_path / "test.ini"
        ini.write_text(
            "[ISOLATION_FOREST]\n"
            f"ModelPath = {pkl}\n"
            f"CalibrationPath = {cal}\n"
            "Scenario = nominal_ops\n"
            f"WriteSideFile = {write_side}\n"
            f"SideFileOutputDir = {tmp_path}\n"
            f"SideFileFlushEvery = {flush_every}\n"
            f"HeartbeatEvery = {heartbeat}\n"
        )
        monkeypatch.setenv("ONAIR_INI_FILE", str(ini))
        return IF_Plugin(MagicMock(), ["foo", "bar"])

    return _build


def _read_side_file(tmp_path):
    files = glob.glob(os.path.join(str(tmp_path), "iforest_out_*.csv"))
    assert len(files) <= 1, f"expected at most one side file, got {files}"
    if not files:
        return None, []
    with open(files[0]) as f:
        rows = list(csv.reader(f))
    return files[0], rows


def test_side_file_path_includes_pid_and_timestamp(configured, tmp_path):
    plugin = configured()
    assert plugin._side_file_path is not None
    name = os.path.basename(plugin._side_file_path)
    assert name.startswith("iforest_out_")
    assert name.endswith(f"_pid{os.getpid()}.csv")


def test_writes_header_and_rows_after_flush_threshold(configured, tmp_path):
    plugin = configured(score=0.1, threshold=0.0, flush_every=2)

    plugin.update(low_level_data=["1.0", "2.0"])  # score 0.1, nominal
    # Single row buffered; nothing on disk yet.
    path, rows = _read_side_file(tmp_path)
    assert path is None

    plugin.update(low_level_data=["3.0", "4.0"])  # second row → flush
    path, rows = _read_side_file(tmp_path)
    assert path is not None
    assert rows[0] == ["frame_idx", "scenario", "score", "threshold",
                       "is_anomaly", "alert", "cleared"]
    assert len(rows) == 3  # header + 2 data rows
    assert rows[1][:2] == ["0", "nominal_ops"]
    assert rows[2][:2] == ["1", "nominal_ops"]
    # is_anomaly / alert / cleared all 0 on a steady nominal stream.
    assert rows[1][4:] == ["0", "0", "0"]
    assert rows[2][4:] == ["0", "0", "0"]


def test_alert_and_clear_bits_track_transitions(configured, tmp_path):
    # Build with score=-0.1 so every frame is anomalous, then flip to 0.1.
    plugin = configured(score=-0.1, threshold=0.0, flush_every=1)

    plugin.update(low_level_data=["1.0", "2.0"])   # frame 1: was_anomaly=False → True (ALERT)
    plugin.update(low_level_data=["1.0", "2.0"])   # frame 2: still anomaly, no edge
    plugin.model.score = 0.1                        # flip back to nominal
    plugin.update(low_level_data=["1.0", "2.0"])   # frame 3: was=True → False (CLEAR)
    plugin.update(low_level_data=["1.0", "2.0"])   # frame 4: nominal steady-state

    _, rows = _read_side_file(tmp_path)
    data = rows[1:]  # drop header
    assert len(data) == 4
    # columns: frame_idx, scenario, score, threshold, is_anomaly, alert, cleared
    is_anom = [r[4] for r in data]
    alerts = [r[5] for r in data]
    cleareds = [r[6] for r in data]
    assert is_anom == ["1", "1", "0", "0"]
    assert alerts == ["1", "0", "0", "0"]
    assert cleareds == ["0", "0", "1", "0"]


def test_disable_switch_suppresses_side_file(configured, tmp_path):
    plugin = configured(write_side="false")
    assert plugin._side_file_path is None
    plugin.update(low_level_data=["1.0", "2.0"])
    plugin.update(low_level_data=["3.0", "4.0"])
    files = glob.glob(os.path.join(str(tmp_path), "iforest_out_*.csv"))
    assert files == []


def test_empty_frame_does_not_advance_or_write(configured, tmp_path):
    """An empty low_level_data is the early-return path; it must not buffer or
    flush. (The IF plugin currently aborts before incrementing the counter, so
    side-file frame_idx stays in lockstep with csv_output rows.)"""
    plugin = configured(flush_every=1)
    plugin.update(low_level_data=[])
    path, _ = _read_side_file(tmp_path)
    assert path is None
    assert plugin._side_file_buffer == []
    assert plugin._frame_count == 0
