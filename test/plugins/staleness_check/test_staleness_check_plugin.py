# GSC-19165-1 OnAIR — staleness_check plugin tests
import os, io, contextlib, tempfile
import pytest

from plugins.staleness_check.staleness_check_plugin import Plugin, _to_num

_TEST_INI = os.path.join(tempfile.gettempdir(), "staleness_check_test.ini")
with open(_TEST_INI, "w") as _f:
    _f.write("[STALENESS_CHECK]\nWriteSideFile=false\nWriteIncidentFile=false\n"
             "SettleFrames=0\nWarmupFrames=15\nStaleThreshold=5\n")

WIDE = "IMU.DeviceHK.DeviceCounter"   # wide uint32 counter, climbs → watched
EVENT = "CFE_SB.MsgSendErrorCounter"  # event tally → excluded by name
U8 = "IMU.DeviceCount"                # uint8 (max<=255) → excluded by wide filter
H = [WIDE, EVENT, U8, "ADCS_GNC.Mode"]


def _mk(headers):
    os.environ["ONAIR_INI_FILE"] = _TEST_INI
    with contextlib.redirect_stdout(io.StringIO()):
        p = Plugin("staleness_check", headers)
    p._side_file_path = None
    p._incident_file_path = None
    return p


def _feed(p, wide, mode="2"):
    row = {WIDE: str(wide), EVENT: "5", U8: "100", "ADCS_GNC.Mode": mode}
    lld = [row.get(h, "[0]") for h in H]
    with contextlib.redirect_stdout(io.StringIO()):
        p.update(lld, {})


def _warmup(p, start=1000):
    for k in range(16):
        _feed(p, wide=start + 5 * k)      # climbs every frame → advances, wide


def test_to_num_sentinel():
    assert _to_num("[0]") is None and _to_num("4") == 4.0


def test_auto_discovers_wide_live_counter():
    p = _mk(H)
    _warmup(p)
    assert H.index(WIDE) in p._watched


def test_excludes_event_counter():
    p = _mk(H)
    _warmup(p)
    assert H.index(EVENT) not in p._watched, "error/event tallies are not liveness"


def test_excludes_uint8_counter():
    p = _mk(H)
    _warmup(p)
    assert H.index(U8) not in p._watched, "uint8 (max<=255) wraps → excluded"


def test_frozen_double_buffer_oscillation_flags():
    # THE key case: a frozen MID's double-buffer alternates between its two last
    # STALE values — the max stops advancing though the value keeps changing.
    p = _mk(H)
    _warmup(p)                                  # max ~1075
    hi, lo = 1075, 1068
    for k in range(12):
        _feed(p, wide=(hi if k % 2 == 0 else lo))   # oscillate two static values
    assert any(WIDE in f for f, _, _ in p._latest_stale), \
        "a frozen oscillating counter (max not advancing) must be flagged"


def test_live_climb_with_flicker_never_stale():
    # live double-buffer: alternates between two CLIMBING values → max advances
    p = _mk(H)
    _warmup(p)
    v = 1075
    for k in range(20):
        v += 6 if k % 2 == 0 else 4             # both buffers climb
        _feed(p, wide=v)
        assert not p._latest_stale


def test_freeze_emits_ex0012_02_incident():
    p = _mk(H)
    _warmup(p)
    closed = []
    p._write_incident = lambda inc: closed.append(inc)
    for _ in range(10):
        _feed(p, wide=1075)                     # frozen (constant → max pinned)
    for k in range(4):
        _feed(p, wide=2000 + k)                 # resumes climbing → closes incident
    assert closed, "a freeze should produce a closed incident"
    assert closed[0].cluster == "EX-0012.02" and WIDE in closed[0].sub_technique
    assert closed[0].mode == "SUNSAFE"
