# GSC-19165-1 OnAIR — session directory tests
import contextlib
import datetime as dt
import io
import json
import os
from unittest.mock import MagicMock

import pytest

from onair.src.util import session as S

NOW = dt.datetime(2026, 10, 1, 12, 0, 0, tzinfo=dt.timezone.utc)


def _iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A sessions root wired through a real ini, with module state reset around each test."""
    sessions = tmp_path / "sessions"
    ini = tmp_path / "onair.ini"
    ini.write_text(
        "[SESSION]\n"
        f"SessionsDir = {sessions}\n"
        "[RULE_GATE]\nWriteSideFile = true\nWriteIncidentFile = true\n"
        f"SideFileOutputDir = {tmp_path / 'legacy'}\n"
        "[CONSISTENCY_CHECK]\nWriteSideFile = true\nWriteIncidentFile = true\n"
        f"SideFileOutputDir = {tmp_path / 'legacy'}\n"
        "[STALENESS_CHECK]\nWriteSideFile = true\nWriteIncidentFile = true\n"
        f"SideFileOutputDir = {tmp_path / 'legacy'}\n"
    )
    monkeypatch.setenv("ONAIR_INI_FILE", str(ini))
    S._reset_for_tests()
    yield sessions
    S._reset_for_tests()


def _mint(root, sid, minted=NOW, **extra):
    os.makedirs(root, exist_ok=True)
    rec = {"session_id": sid, "purpose": "attack", "technique": "DE-0003.01",
           "mode": "INERTIAL", "status": "new", "minted_utc": _iso(minted)}
    rec.update(extra)
    (root / S.CLAIM_NAME).write_text(json.dumps(rec))


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ── claiming ─────────────────────────────────────────────────────────────────────

def test_inert_without_sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ONAIR_INI_FILE", str(tmp_path / "missing.ini"))
    S._reset_for_tests()
    try:
        assert S.claim() is None
        assert S.current() is None
        assert S.path("raw") is None
    finally:
        S._reset_for_tests()


def test_claim_renames_the_harness_file_into_the_session(root):
    sid = "2026-10-01T11-59-59Z_attack_DE-0003.01_INERTIAL"
    _mint(root, sid)
    s = _quiet(S.claim, now=NOW)
    assert s.id == sid
    assert not (root / S.CLAIM_NAME).exists()            # claimed exactly once
    rec = json.loads((root / sid / "session.json").read_text())
    assert rec["technique"] == "DE-0003.01" and rec["mode"] == "INERTIAL"   # harness fields kept
    assert rec["onair"]["claim"] == "harness" and rec["onair"]["pid"] == os.getpid()
    assert rec["started_utc"] == _iso(NOW)
    assert (root / S.CURRENT_NAME).read_text().strip() == sid
    assert _quiet(S.claim) is s                           # idempotent


def test_no_claim_file_gives_an_adhoc_session(root):
    s = _quiet(S.claim, now=NOW)
    assert s.id == "2026-10-01T12-00-00Z_adhoc"
    rec = s.read()
    assert rec["purpose"] == "adhoc" and rec["status"] == "new"
    assert rec["onair"]["claim"] == "adhoc" and "no claim file" in rec["onair"]["claim_skipped"]


def test_stale_claim_is_ignored_and_left_in_place(root):
    _mint(root, "2026-10-01T10-00-00Z_soak_SUNSAFE", minted=NOW - dt.timedelta(seconds=S.CLAIM_MAX_AGE_S + 1))
    s = _quiet(S.claim, now=NOW)
    assert s.id.endswith("_adhoc")
    assert "stale" in s.read()["onair"]["claim_skipped"]
    assert (root / S.CLAIM_NAME).exists()
    assert not (root / "2026-10-01T10-00-00Z_soak_SUNSAFE").exists()


def test_claim_never_reuses_an_existing_session_directory(root):
    sid = "2026-10-01T11-59-59Z_soak_BDOT"
    (root / sid).mkdir(parents=True)
    (root / sid / "session.json").write_text('{"session_id": "%s", "precious": true}' % sid)
    _mint(root, sid)
    s = _quiet(S.claim, now=NOW)
    assert s.id.endswith("_adhoc")
    assert json.loads((root / sid / "session.json").read_text())["precious"] is True


def test_two_adhoc_sessions_in_one_second_do_not_collide(root):
    (root / "2026-10-01T12-00-00Z_adhoc").mkdir(parents=True)
    assert _quiet(S.claim, now=NOW).id == "2026-10-01T12-00-00Z_adhoc-2"


def test_update_deep_merges_and_is_atomic(root):
    s = _quiet(S.claim, now=NOW)
    s.update({"onair": {"blend_mode": "tap"}, "files": {"raw": "raw.csv"}})
    s.update({"files": {"blended": "blended.csv"}})
    rec = s.read()
    assert rec["onair"]["claim"] == "adhoc" and rec["onair"]["blend_mode"] == "tap"
    assert rec["files"] == {"raw": "raw.csv", "blended": "blended.csv"}
    assert not [p for p in os.listdir(s.dir) if ".tmp" in p]


# ── fixed file names ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key,rel", [
    ("iforest", "detectors/iforest.csv"),
    ("attack_class", "detectors/attack_class.csv"),
    ("incident", "detectors/incident.csv"),
    ("iforest_golden", "detectors/iforest_golden.npz"),
])
def test_side_file_is_a_fixed_name_in_the_session(root, tmp_path, key, rel):
    s = _quiet(S.claim, now=NOW)
    ext = "npz" if key == "iforest_golden" else "csv"
    got = S.side_file(key, str(tmp_path / "legacy"), "legacy_stem", ext=ext)
    assert got == os.path.join(s.dir, rel)
    assert not (tmp_path / "legacy").exists()             # the legacy dir is not even created


def test_side_file_falls_back_to_the_legacy_name_without_a_session(tmp_path, monkeypatch):
    monkeypatch.setenv("ONAIR_INI_FILE", str(tmp_path / "missing.ini"))
    S._reset_for_tests()
    try:
        got = S.side_file("iforest", str(tmp_path / "csv"), "iforest_out")
        assert os.path.dirname(got) == str(tmp_path / "csv")
        assert os.path.basename(got).startswith("iforest_out_") and got.endswith(f"_pid{os.getpid()}.csv")
    finally:
        S._reset_for_tests()


@pytest.mark.parametrize("module,name,out_key,inc_key", [
    ("plugins.rule_gate.rule_gate_plugin", "rule_gate", "rule_gate", "rule_gate_incident"),
    ("plugins.consistency_check.consistency_check_plugin", "consistency_check",
     "consistency", "consistency_incident"),
    ("plugins.staleness_check.staleness_check_plugin", "staleness_check",
     "staleness", "staleness_incident"),
])
def test_gate_plugins_write_into_the_session(root, module, name, out_key, inc_key):
    import importlib
    s = _quiet(S.claim, now=NOW)
    Plugin = importlib.import_module(module).Plugin
    p = _quiet(Plugin, name, ["ADCS_GNC.Mode", "CFE_EVS_HK.MessageSendCounter"])
    assert p._side_file_path == os.path.join(s.dir, "detectors", f"{out_key}.csv")
    assert p._incident_file_path == os.path.join(s.dir, "detectors", f"{inc_key}.csv")


def test_csv_output_writes_raw_csv_and_records_the_schema(root, tmp_path):
    from plugins.csv_output.csv_output_plugin import Plugin as CSV_Output
    s = _quiet(S.claim, now=NOW)
    out = CSV_Output(MagicMock(), ["h1", "h2"])
    out.update(["a", "b"], {})
    out.render_reasoning()
    out.update(["c", "d"], {})
    out.render_reasoning()
    assert out.file_name == os.path.join(s.dir, "raw.csv")
    assert open(out.file_name).read().splitlines() == ["h1,h2", "a,b", "c,d"]
    meta = json.loads(open(os.path.join(s.dir, "raw.meta.json")).read())
    rec = s.read()
    assert rec["files"]["raw"] == "raw.csv" and rec["files"]["raw_meta"] == "raw.meta.json"
    assert rec["schema"]["recorded_sha256"] == meta["recorded_schema_sha256"]
    assert rec["schema"]["n_cols"] == 2
    assert not os.path.exists(tmp_path / "data")          # no legacy data/csv created


def test_blended_writer_and_arrival_meter_use_fixed_names(root):
    import sys
    sys.modules.setdefault("sbn_python_client", MagicMock())
    sys.modules.setdefault("message_headers", MagicMock())
    from onair.data_handling.sbn_adapter_blended import BlendedCsvWriter, ArrivalMeter
    s = _quiet(S.claim, now=NOW)
    opened = []
    w = BlendedCsvWriter(["a", "b"], "/nonexistent/should/not/be/created", "x_{pid}", set(),
                         fixed_path=s.path("blended"),
                         on_open=lambda *a: opened.append(a))
    w.write(["1", "2"])
    w.close()
    assert os.path.exists(os.path.join(s.dir, "blended.csv"))
    assert os.path.exists(os.path.join(s.dir, "blended.meta.json"))
    assert opened and opened[0][0] == os.path.join(s.dir, "blended.csv") and opened[0][3] == 2
    assert not os.path.exists("/nonexistent")

    m = ArrivalMeter(1, "/nonexistent/arrivals", clock=lambda: 0.0, fixed_path=s.path("arrivals"))
    m.write({"utc": "t", "window_s": 1.0, "packets": {}})
    assert m.file_name == os.path.join(s.dir, "arrivals.jsonl")
    assert json.loads(open(m.file_name).read())["window_s"] == 1.0
