# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""AINOS3-126 — the live blend must equal the offline one.

The load-bearing tests here are the EQUIVALENCE ones. The adapter is easy; the
deliverable is the evidence that `deinterleave_csv.py(raw) == native blend`,
because if the two ever disagree, offline results stop predicting live
behaviour and nothing in the system would notice.

Two further tests exist purely as anti-drift pins. `_stringify` and the CSV
writer are deliberately DUPLICATED from `csv_output_plugin` (the adapter loads
inside the OnAIR core package, before any plugin path is resolved), so these
assert the copies still agree with the originals.
"""

import csv
import importlib.util
import os
import sys
from unittest.mock import MagicMock

import pytest

# mock dependencies of sbn_adapter.py
sys.modules.setdefault('sbn_python_client', MagicMock())
sys.modules.setdefault('message_headers', MagicMock())

import onair.data_handling.sbn_adapter_blended as blended
from onair.data_handling.sbn_adapter_blended import (
    SENTINEL, SIM_TIME_COLUMN, TIMESTAMP_COLUMN, BlendEngine, BlendedCsvWriter,
    DataSource, SimClock, _stringify, load_adapter_config, read_sim_epoch,
    resolve_blended_dir)


# ---------------------------------------------------------------- helpers

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONAIR = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..'))


def _load_by_path(mod_name, rel_path):
    path = os.path.join(_ONAIR, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def offline():
    """training/deinterleave_csv.py — the reference implementation."""
    return _load_by_path("deinterleave_csv_ref", "training/deinterleave_csv.py")


def _offline_blend(offline, frames, fields):
    """Run the reference transform over stringified frames."""
    rows = [{f: _stringify(v) for f, v in zip(fields, fr)} for fr in frames]
    out, _stats = offline.deinterleave(rows, fields)
    return [[r[f] for f in fields] for r in out]


def _native_blend(frames, fields, start_buf=0):
    eng = BlendEngine(len(fields))
    out = []
    for i, fr in enumerate(frames):
        bs, _bo = eng.feed(fr, (start_buf + i) % 2)
        out.append(list(bs))
    return out


# ---------------------------------------------------------------- BlendEngine

def test_blend_engine_seeds_every_field_with_the_sentinel():
    eng = BlendEngine(4)
    assert eng._blend_s == [SENTINEL] * 4
    # The object seed must stringify to the same sentinel, so the two
    # representations agree even for a field nothing ever delivers.
    assert [_stringify(o) for o in eng._blend_o] == [SENTINEL] * 4


def test_blend_engine_first_frame_from_a_buffer_adopts_everything_it_carries():
    eng = BlendEngine(3)
    bs, _ = eng.feed(["1", "2", SENTINEL], 0)
    # The sentinel field is NOT adopted; it stays seeded.
    assert bs == ["1", "2", SENTINEL]


def test_blend_engine_carries_a_field_forward_when_neither_buffer_updates_it():
    eng = BlendEngine(2)
    eng.feed(["a", "1"], 0)
    eng.feed(["b", "1"], 1)
    bs, _ = eng.feed(["a", "9"], 0)     # col0 unchanged vs buffer 0's own ref
    # col0 keeps buffer 1's fresher "b"; only col1 is news.
    assert bs == ["b", "9"]


def test_blend_engine_does_not_let_a_stale_buffer_overwrite_a_fresh_update():
    """The whole point: a naive 'latest value' dict reproduces the flicker."""
    eng = BlendEngine(1)
    eng.feed(["old"], 0)        # buffer 0 ref = old
    eng.feed(["old"], 1)        # buffer 1 ref = old
    bs, _ = eng.feed(["new"], 0)
    assert bs == ["new"]
    # Buffer 1 still holds "old" but has NOT changed against its own ref, so it
    # is not news and must not undo buffer 0's update.
    bs, _ = eng.feed(["old"], 1)
    assert bs == ["new"]


def test_blend_engine_never_adopts_the_sentinel_over_good_telemetry():
    eng = BlendEngine(1)
    eng.feed(["7"], 0)
    eng.feed([SENTINEL], 1)     # buffer 1 has never received this MID
    bs, _ = eng.feed([SENTINEL], 1)
    assert bs == ["7"]


def test_blend_engine_writes_the_sentinel_for_a_field_no_buffer_delivers():
    """⚠ Blank is NOT equivalent: four corpus tools key on the literal [0] and
    two of them fail silently on "" (the AINOS3-100 capture gate and the
    AINOS3-86 FP figure)."""
    eng = BlendEngine(2)
    eng.feed(["5", SENTINEL], 0)
    bs, _ = eng.feed(["5", SENTINEL], 1)
    assert bs[1] == SENTINEL
    assert bs[1] != ""


def test_blend_engine_objects_track_the_strings():
    eng = BlendEngine(2)
    eng.feed([[1, 2], "x"], 0)
    bs, bo = eng.feed([[1, 2], "y"], 1)
    assert [_stringify(o) for o in bo] == list(bs)
    # inline mode must hand the pipeline the original types, not strings.
    assert bo[0] == [1, 2]


def test_blend_engine_stats_count_updates_per_buffer():
    eng = BlendEngine(2)
    eng.feed(["a", "b"], 0)
    eng.feed(["a", "b"], 1)
    eng.feed(["a", "c"], 0)
    st = eng.stats()
    assert st["frames"] == 3
    assert st["updates_buf0"] == 1
    assert st["updates_buf1"] == 0
    assert st["first_seen"] == [True, True]


# ------------------------------------------------- equivalence with offline

def test_native_blend_equals_offline_blend_on_a_synthetic_stream(offline):
    fields = [f"c{i}" for i in range(6)]
    frames = [
        ["1", "a", SENTINEL, "0", "x", SENTINEL],
        ["1", "b", SENTINEL, "0", "x", SENTINEL],
        ["2", "a", SENTINEL, "0", "y", SENTINEL],
        ["1", "b", "9", "0", "x", SENTINEL],
        ["2", "c", "9", "1", "y", SENTINEL],
        ["3", "b", "9", "0", "x", SENTINEL],
    ]
    assert _native_blend(frames, fields) == _offline_blend(offline, frames, fields)


def test_native_blend_equals_offline_blend_on_a_randomised_stream(offline):
    """Fuzz the shapes the real stream actually has: fields that alternate every
    frame, fields that update rarely, and fields that never arrive at all."""
    rng = pytest.gen
    fields = [f"c{i}" for i in range(40)]
    state = [[SENTINEL] * 40, [SENTINEL] * 40]
    frames = []
    for i in range(300):
        b = i % 2
        row = list(state[b])
        for j in range(40):
            if j % 10 == 0:
                continue                       # never delivered: stays sentinel
            if rng.random() < (0.9 if j % 3 == 0 else 0.05):
                row[j] = str(rng.randint(0, 5))
        state[b] = row
        frames.append(list(row))
    assert _native_blend(frames, fields) == _offline_blend(offline, frames, fields)


def test_parity_label_symmetry(offline):
    """⚠ The offline script calls CSV row 0 'buffer 0'; live, row 0 comes from
    double_buffer_read_index == 1. The blend is symmetric under swapping the two
    slots, so the labelling difference must not change the output."""
    fields = [f"c{i}" for i in range(5)]
    frames = [[str(i % 3), str(i), "k", SENTINEL, str(i // 2)] for i in range(50)]
    assert _native_blend(frames, fields, start_buf=0) == \
           _native_blend(frames, fields, start_buf=1)


def test_blend_is_per_field_independent_so_pruning_commutes(offline):
    """csv_output prunes ExcludeColumns AFTER the blend; the offline script
    blends the ALREADY-pruned CSV. That is only sound if the transform is
    per-field independent."""
    fields = [f"c{i}" for i in range(6)]
    frames = [[str((i + j) % 4) for j in range(6)] for i in range(40)]
    full = _native_blend(frames, fields)
    keep = [0, 2, 5]
    pruned_after = [[row[k] for k in keep] for row in full]
    pruned_first = _native_blend([[fr[k] for k in keep] for fr in frames],
                                 [fields[k] for k in keep])
    assert pruned_after == pruned_first


# ------------------------------------------------------ anti-drift pins

def test_stringify_matches_csv_output():
    """⚠ `_stringify` is duplicated from csv_output_plugin. If they diverge, the
    blend decides what changed in a different domain than the CSV records, and
    the equivalence proof silently stops meaning anything."""
    co = _load_by_path("csv_output_plugin_ref", "fsw/plugins/csv_output/csv_output_plugin.py")
    cases = [0, 1, -3, 0.0, 1.5, "x", "", "[0]", [0], [1, 2], [[1, 2], [3]],
             b"abc\x00\x00", b"", True, None, "Events squelched, AppName = LC"]
    for c in cases:
        assert _stringify(c) == co._stringify(c), f"diverged on {c!r}"


def test_writer_matches_csv_output_byte_for_byte(tmp_path, mocker):
    """⚠ BlendedCsvWriter is mirrored from csv_output's Plugin. Same pruning,
    same QUOTE_MINIMAL quoting, same header row — pinned here because the
    blended file has to be comparable to its raw sibling cell for cell."""
    co = _load_by_path("csv_output_plugin_ref2", "fsw/plugins/csv_output/csv_output_plugin.py")
    headers = ["a", "b", "drop", "c"]
    excl = {"drop"}
    # EVS messages routinely carry embedded commas and quotes — the case that
    # makes naive ','.join() writers wrong.
    frames = [["1", 'Events squelched, AppName = LC', "z", '"q"'],
              ["2", "plain", "z", "[0]"]]

    mine = BlendedCsvWriter(headers, str(tmp_path), "mine", excl)
    for fr in frames:
        mine.write(fr)
    mine.close()

    theirs = co.Plugin.__new__(co.Plugin)
    theirs.headers = list(headers)
    theirs.exclude_columns = set(excl)
    theirs.output_dir = str(tmp_path)
    theirs.filename_template = "theirs"
    theirs.lines_per_file = 0
    theirs.write_header_on_rotation = True
    theirs.pid = os.getpid()
    theirs.headers_built = True
    theirs.headers_written_to_current_file = False
    theirs.lines_current = 0
    theirs.current_buffer = []
    theirs.file_name = None
    theirs.keep_indices = None
    theirs.filtered_headers = None
    theirs._meta_extras = {}
    for fr in frames:
        theirs.update(low_level_data=fr)
        theirs.render_reasoning()

    assert open(mine.file_name, "rb").read() == open(theirs.file_name, "rb").read()
    assert mine.recorded_schema_sha256() == theirs._recorded_schema_sha256()


# ------------------------------------------------------------ writer

def test_writer_prunes_exclude_columns_and_writes_a_sidecar(tmp_path):
    w = BlendedCsvWriter(["a", "b", "c"], str(tmp_path), "blend_{pid}", {"b"},
                         meta_extras={"transform": "t"})
    w.write(["1", "2", "3"])
    w.close()
    rows = list(csv.reader(open(w.file_name)))
    assert rows == [["a", "c"], ["1", "3"]]
    import json
    meta = json.load(open(w.file_name[:-4] + ".meta.json"))
    assert meta["kept_columns"] == ["a", "c"]
    assert meta["excluded_columns"] == ["b"]
    assert meta["transform"] == "t"
    assert meta["recorded_schema_sha256"] == w.recorded_schema_sha256()


def test_writer_creates_no_file_until_the_first_row(tmp_path):
    w = BlendedCsvWriter(["a"], str(tmp_path), "blend_{pid}", set())
    assert os.listdir(tmp_path) == []
    w.write(["1"])
    w.close()
    assert len(os.listdir(tmp_path)) == 2      # csv + sidecar


# ------------------------------------------------------------ config

def test_load_adapter_config_defaults_to_tap_when_no_ini(tmp_path):
    cfg = load_adapter_config(str(tmp_path / "missing.ini"))
    assert cfg["blendmode"] == "tap"


def test_load_adapter_config_shares_outputdir_and_excludes_with_csv_output(tmp_path):
    ini = tmp_path / "x.ini"
    ini.write_text(
        "[CSV_OUTPUT]\n"
        "OutputDir = /tmp/somewhere\n"
        "FilenameTemplate = csv_out_{timestamp}_pid{pid}\n"
        "ExcludeColumns = a,b , c\n"
        "[SBN_ADAPTER]\n"
        "BlendMode = inline\n"
        "ProfileEvery = 500\n"
    )
    cfg = load_adapter_config(str(ini))
    assert cfg["blendmode"] == "inline"
    assert cfg["outputdir"] == "/tmp/somewhere"
    assert cfg["excludecolumns"] == "a,b , c"
    assert cfg["profileevery"] == "500"


def test_blended_file_keeps_the_csv_out_basename():
    """⚠ The DIRECTORY marks data as blended, not the filename. That is the
    convention deinterleave_csv.py set (it preserves the source basename into
    --out-dir), so every tool globbing csv_out_*.csv — loader.load,
    list_clean_csvs, build_corpus_manifest — works on either directory with no
    filename special case."""
    cfg = load_adapter_config(os.devnull)
    assert cfg["blendedfilenametemplate"] == "csv_out_{timestamp}_pid{pid}"


def test_blended_dir_defaults_to_a_sibling_not_csv_outputs_own_dir():
    """⚠ NOT data/onair/csv. That directory already mixes 11 recording
    generations with every plugin side-file, and AINOS3-125 AC4 put the
    offline-blended corpus in data/onair/csv_blended/ — native output has to
    land in the same place or one corpus is split across two conventions."""
    assert resolve_blended_dir(None, "../../../../data/onair/csv") == \
        "../../../../data/onair/csv_blended"
    assert resolve_blended_dir("", "/a/b/csv") == "/a/b/csv_blended"
    assert resolve_blended_dir("   ", "csv") == "csv_blended"      # normpath, not ./


def test_blended_dir_honours_an_explicit_override():
    assert resolve_blended_dir("/elsewhere", "/a/b/csv") == "/elsewhere"


def test_side_file_lands_in_its_own_directory(mocker, tmp_path):
    raw_dir = tmp_path / "csv"
    raw_dir.mkdir()
    # blendedoutputdir None on purpose: this test covers the DEFAULT resolution,
    # which the _cut fixture otherwise overrides for per-test isolation.
    cut = _cut("tap", ["a"], mocker, raw_dir,
               cfg_extra={"outputdir": str(raw_dir), "blendedoutputdir": None})
    mocker.patch.object(blended.sbn_adapter.DataSource, 'get_next', return_value=["1"])
    cut.get_next()
    cut._writer.close()
    assert os.listdir(raw_dir) == []                       # raw dir untouched
    assert sorted(os.listdir(tmp_path / "csv_blended")) == [
        os.path.basename(cut._writer.file_name),
        os.path.basename(cut._writer.file_name)[:-4] + ".meta.json"]


# ---------------------------------------------------- per-row timestamp

def _meta_cut(mocker, *, stamp=True, n=3):
    """A DataSource whose parent parse_meta_data_file is stubbed out."""
    cut = DataSource.__new__(DataSource)
    cut._stamp_frames = stamp
    cut._ts_idx = None
    cut._stamp_simtime = False
    cut._sim_idx = None
    cut._sim_src = None
    cut._sim_clock = None
    cut.currentData = [{"headers": [f"c{i}" for i in range(n)],
                        "data": [[0]] * n} for _ in range(2)]
    cfg = {"data_labels": [f"c{i}" for i in range(n)],
           "subsystem_assignments": [["MISSION"]] * n,
           "test_assignments": [[["NOOP"]]] * n,
           "description_assignments": ["d"] * n}
    mocker.patch.object(blended.sbn_adapter.DataSource, 'parse_meta_data_file',
                        return_value=cfg)
    return cut, cfg


def test_timestamp_column_is_appended_to_every_binning_list(mocker):
    """⚠ vehicle_rep.py:24 asserts len(headers) == len(tests), so the parallel
    binning lists must grow with the labels, not just data_labels."""
    cut, _ = _meta_cut(mocker)
    cfg = cut.parse_meta_data_file("meta.json", False)
    assert cfg["data_labels"][-1] == TIMESTAMP_COLUMN
    n = len(cfg["data_labels"])
    assert len(cfg["subsystem_assignments"]) == n
    assert len(cfg["test_assignments"]) == n
    assert len(cfg["description_assignments"]) == n
    # and the frame layout grew to match, in BOTH buffers
    for buf in cut.currentData:
        assert buf["headers"][-1] == TIMESTAMP_COLUMN
        assert len(buf["data"]) == n


def test_timestamp_column_can_be_switched_off(mocker):
    cut, _ = _meta_cut(mocker, stamp=False)
    cfg = cut.parse_meta_data_file("meta.json", False)
    assert TIMESTAMP_COLUMN not in cfg["data_labels"]
    assert cut._ts_idx is None


def test_get_next_stamps_the_returned_frame(mocker, tmp_path):
    import datetime as _dt
    cut = _cut("tap", ["a", "b", TIMESTAMP_COLUMN], mocker, tmp_path)
    cut._ts_idx = 2
    mocker.patch.object(blended.sbn_adapter.DataSource, 'get_next',
                        return_value=["1", "2", ""])
    out = cut.get_next()
    cut._writer.close()
    stamped = out[2]
    parsed = _dt.datetime.fromisoformat(stamped)     # manifest start_utc form
    assert parsed.tzinfo is not None
    # the SAME value reaches the blended file — one clock reading, two records
    rows = list(csv.reader(open(cut._writer.file_name)))
    assert rows[1][2] == stamped


def test_timestamp_is_adopted_by_the_blend_on_every_row(mocker, tmp_path):
    """It differs from its own buffer's previous frame every time, so the
    change-detection rule adopts it on every row rather than carrying a stale
    one forward."""
    cut = _cut("tap", ["a", TIMESTAMP_COLUMN], mocker, tmp_path)
    cut._ts_idx = 1
    mocker.patch.object(blended.sbn_adapter.DataSource, 'get_next',
                        side_effect=[["x", ""], ["x", ""], ["x", ""]])
    seen = []
    for i in range(3):
        cut.double_buffer_read_index = (i + 1) % 2
        seen.append(cut.get_next()[1])
    cut._writer.close()
    rows = list(csv.reader(open(cut._writer.file_name)))[1:]
    assert [r[1] for r in rows] == seen
    assert len(set(seen)) == 3          # never a stale carry-forward


# ---------------------------------------------------- simulation time

import datetime as _dt  # noqa: E402

_EPOCH = _dt.datetime(2025, 10, 20, 17, 43, 20, tzinfo=_dt.timezone.utc)
# Real values recorded 2026-09-21; week 341 is the 10-bit field.
_W, _SOW, _LEAP = "341", "150240", "37"


def test_sim_clock_resolves_the_gps_week_rollover_against_the_configured_epoch():
    """⚠ Without the rollover correction week 341 decodes to 1986-07-21, not
    2025-10-20 — and the TIME OF DAY is right either way, which is what makes
    the bug survivable and worth pinning."""
    got = SimClock(_EPOCH).update(_W, _SOW, _LEAP)
    t = _dt.datetime.fromisoformat(got)
    assert t.year == 2025 and t.month == 10 and t.day == 20
    assert abs((t - _EPOCH).total_seconds()) < 60


def test_sim_clock_is_blank_until_it_can_resolve_the_era():
    """Better an empty cell than a date wrong by a multiple of ~19.6 years."""
    assert SimClock(None).update(_W, _SOW, _LEAP) == ""


def test_sim_clock_rejects_torn_reads_and_holds_the_last_good_value():
    """NOVATEL torn reads step backwards; measured 10 per 2,610 frames."""
    c = SimClock(_EPOCH)
    first = c.update(_W, _SOW, _LEAP)
    second = c.update(_W, str(float(_SOW) + 10), _LEAP)
    torn = c.update(_W, str(float(_SOW) - 500), _LEAP)      # backwards
    jump = c.update(_W, str(float(_SOW) + 99999), _LEAP)    # absurd forward
    assert first < second == torn == jump
    assert c.rejected == 2


def test_sim_clock_ignores_unparseable_and_out_of_range_fields():
    c = SimClock(_EPOCH)
    assert c.update("[0]", "[0]", "[0]") == ""
    assert c.update("-1", _SOW, _LEAP) == ""
    assert c.update(_W, "999999999", _LEAP) == ""
    good = c.update(_W, _SOW, _LEAP)
    assert good and _dt.datetime.fromisoformat(good).year == 2025


def test_read_sim_epoch_parses_42s_own_input_file(tmp_path, monkeypatch):
    """Read from cfg, never hard-coded — changing the epoch to test another
    orbital phase must not need a code change."""
    d = tmp_path / "cfg" / "InOut"
    d.mkdir(parents=True)
    (d / "Inp_Sim.txt").write_text(
        "NOS3   ! Time Mode\n"
        "03 14 2027    !  Date (UTC) (Month, Day, Year)\n"
        "09 26 53.00   !  Time (UTC) (Hr,Min,Sec)\n")
    monkeypatch.chdir(tmp_path)
    assert read_sim_epoch() == _dt.datetime(2027, 3, 14, 9, 26, 53,
                                            tzinfo=_dt.timezone.utc)


def test_read_sim_epoch_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert read_sim_epoch() is None


def test_both_derived_columns_are_added_with_their_binning(mocker):
    cut, _ = _meta_cut(mocker)
    cut.currentData[0]["headers"] += ["NOVATEL.Novatel_oem615.Weeks",
                                      "NOVATEL.Novatel_oem615.SecondsIntoWeek",
                                      "CFE_TIME.LeapSeconds"]
    cut.currentData[1]["headers"] = list(cut.currentData[0]["headers"])
    for b in cut.currentData:
        b["data"] = [[0]] * len(b["headers"])
    cut._stamp_simtime = True
    cut._sim_src = None
    cut._sim_clock = None
    cfg = cut.parse_meta_data_file("meta.json", False)
    assert cfg["data_labels"][-2:] == [TIMESTAMP_COLUMN, SIM_TIME_COLUMN]
    n = len(cfg["data_labels"])
    assert len(cfg["test_assignments"]) == n == len(cfg["subsystem_assignments"])


def test_sim_column_skipped_when_gps_is_not_subscribed(mocker, capsys):
    """Do not emit a column that can never be filled."""
    cut, _ = _meta_cut(mocker)
    cut._stamp_simtime = True
    cut._sim_src = None
    cut._sim_clock = None
    cfg = cut.parse_meta_data_file("meta.json", False)
    assert SIM_TIME_COLUMN not in cfg["data_labels"]
    assert cut._stamp_simtime is False
    assert "SKIPPED" in capsys.readouterr().out


# ------------------------------------------------------------ DataSource

def _cut(mode, headers, mocker, tmp_path=None, cfg_extra=None):
    cut = DataSource.__new__(DataSource)
    cut.blend_mode = mode
    cut._blend = None
    cut._writer = None
    cut._frames = 0
    cut._prof = None
    cut._warned_len = False
    cut._stamp_frames = True
    cut._ts_idx = None          # tests that exercise stamping set it explicitly
    cut._stamp_simtime = False
    cut._sim_idx = None
    cut._sim_src = None
    cut._sim_clock = None
    cut.all_headers = list(headers)
    cut.double_buffer_read_index = 0
    cfg = {"excludecolumns": "", "outputdir": str(tmp_path) if tmp_path else "/tmp",
           "blendedfilenametemplate": "csv_out_{pid}",
           "blendedoutputdir": str(tmp_path) if tmp_path else "/tmp",
           "flushevery": "1",
           "schema_path": None, "schema_sha256": None}
    cfg.update(cfg_extra or {})
    cut._cfg = cfg
    return cut


def test_get_next_off_mode_is_the_parent_adapter(mocker, tmp_path):
    cut = _cut("off", ["a", "b"], mocker, tmp_path)
    frame = ["1", "2"]
    mocker.patch.object(blended.sbn_adapter.DataSource, 'get_next', return_value=frame)
    assert cut.get_next() is frame
    assert cut._blend is None
    assert cut._writer is None


def test_get_next_tap_mode_returns_the_raw_frame_and_side_writes_the_blend(mocker, tmp_path):
    """⚠ AC2b: in tap mode the pipeline must receive exactly what the parent
    adapter returns, or old and new csv/ data are not the same representation."""
    cut = _cut("tap", ["a", "b"], mocker, tmp_path)
    frames = [["1", "x"], ["1", "y"], ["2", "x"]]
    mocker.patch.object(blended.sbn_adapter.DataSource, 'get_next',
                        side_effect=[list(f) for f in frames])
    out = []
    for i, _ in enumerate(frames):
        cut.double_buffer_read_index = (i + 1) % 2
        out.append(cut.get_next())
    cut._writer.close()

    assert out == frames                          # raw, untouched
    rows = list(csv.reader(open(cut._writer.file_name)))
    assert rows[0] == ["a", "b"]
    assert rows[1:] == [["1", "x"], ["1", "y"], ["2", "y"]]


def test_get_next_inline_mode_returns_the_blend_and_writes_no_side_file(mocker, tmp_path):
    cut = _cut("inline", ["a", "b"], mocker, tmp_path)
    frames = [["1", "x"], ["1", "y"], ["2", "x"]]
    mocker.patch.object(blended.sbn_adapter.DataSource, 'get_next',
                        side_effect=[list(f) for f in frames])
    out = []
    for i, _ in enumerate(frames):
        cut.double_buffer_read_index = (i + 1) % 2
        out.append(list(cut.get_next()))

    assert cut._writer is None
    assert os.listdir(tmp_path) == []
    assert out[-1] == ["2", "y"]


def test_side_file_disabled_when_headers_and_frame_disagree(mocker, tmp_path, capsys):
    """A length mismatch would mislabel every column in the blended file. Refuse
    to write it rather than emit a file nothing can detect is wrong."""
    cut = _cut("tap", ["a", "b", "c"], mocker, tmp_path)
    mocker.patch.object(blended.sbn_adapter.DataSource, 'get_next', return_value=["1", "2"])
    cut.get_next()
    assert cut._writer is None
    assert cut._blend is not None                 # blending still runs
    assert "ERROR" in capsys.readouterr().out
    assert os.listdir(tmp_path) == []


def test_tap_mode_output_matches_the_offline_blend_of_its_own_raw_file(mocker, tmp_path, offline):
    """End-to-end, the AC2 shape: run frames through the adapter, write BOTH
    files, then blend the raw one offline and require equality."""
    fields = [f"c{i}" for i in range(8)]
    rng = pytest.gen
    state = [[SENTINEL] * 8, [SENTINEL] * 8]
    frames = []
    for i in range(120):
        row = list(state[i % 2])
        for j in range(8):
            if j == 3:
                continue
            if rng.random() < 0.4:
                row[j] = str(rng.randint(0, 9))
        state[i % 2] = row
        frames.append(list(row))

    cut = _cut("tap", fields, mocker, tmp_path)   # blended -> tmp_path, raw -> tmp_path/raw
    mocker.patch.object(blended.sbn_adapter.DataSource, 'get_next',
                        side_effect=[list(f) for f in frames])
    raw_w = BlendedCsvWriter(fields, str(tmp_path / "raw"), "csv_out_{pid}", set())
    for i, _ in enumerate(frames):
        cut.double_buffer_read_index = (i + 1) % 2
        raw_w.write([_stringify(v) for v in cut.get_next()])
    raw_w.close()
    cut._writer.close()

    sys.path.insert(0, os.path.join(_ONAIR, "training"))
    import verify_blend_equivalence as v
    r = v.compare(raw_w.file_name, cut._writer.file_name)
    assert r["ok"], r["diffs"]
    assert r["compared_rows"] == len(frames)
