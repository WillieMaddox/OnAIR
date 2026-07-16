# GSC-19165-1 OnAIR — consistency_check plugin tests
import os, io, contextlib, tempfile
import pytest

from plugins.consistency_check.consistency_check_plugin import Plugin, _to_num

# Real ini disabling file output (construction must not touch the runtime-relative
# data dir, unwritable from the pytest cwd). Tests inspect in-memory state.
_TEST_INI = os.path.join(tempfile.gettempdir(), "consistency_check_test.ini")
with open(_TEST_INI, "w") as _f:
    _f.write("[CONSISTENCY_CHECK]\nWriteSideFile=false\nWriteIncidentFile=false\n")

# A WIDE counter (uint32, discovered), a uint8 counter (excluded by max<=255), a
# physical non-counter field (excluded by name), and mode.
WIDE = "CFE_EVS_HK.MessageSendCounter"
U8 = "IMU.DeviceCount"
PHYS = "MAG_DEV.MagneticIntensityX"
H = [WIDE, U8, PHYS, "ADCS_GNC.Mode"]


def _mk(headers):
    os.environ["ONAIR_INI_FILE"] = _TEST_INI
    with contextlib.redirect_stdout(io.StringIO()):
        p = Plugin("consistency_check", headers)
    p._side_file_path = None
    p._incident_file_path = None
    return p


def _feed(p, wide=None, u8="130", phys="5500", mode="2"):
    row = {U8: u8, PHYS: phys, "ADCS_GNC.Mode": mode}
    if wide is not None:
        row[WIDE] = str(wide)
    lld = [row.get(h, "[0]") for h in H]
    with contextlib.redirect_stdout(io.StringIO()):
        p.update(lld, {})


def _warmup(p, start=10000):
    # WIDE climbs (uint32, discovered); U8 climbs 100..130 (counter-named but ≤255,
    # excluded); PHYS oscillates (not counter-named, never a candidate).
    for k in range(31):
        _feed(p, wide=start + k, u8=str(100 + (k % 30)), phys=str(5500 + (k % 5) * 3))


def test_to_num_handles_sentinel():
    assert _to_num("[0]") is None and _to_num("9") == 9.0


def test_auto_discovers_wide_counter():
    p = _mk(H)
    _warmup(p)
    assert H.index(WIDE) in p._watched


def test_excludes_uint8_counter():
    p = _mk(H)
    _warmup(p)
    assert H.index(U8) not in p._watched, "uint8 (max<=255) counters wrap → excluded"


def test_excludes_non_counter_named_field():
    p = _mk(H)
    _warmup(p)
    assert H.index(PHYS) not in p._watched, "physical fields are not counter-named"


def test_double_buffer_flicker_not_flagged():
    # THE key FP case: a wide counter oscillating between two buffer values.
    p = _mk(H)
    _warmup(p)                       # window filled ~10023..10030
    for lo, hi in [(10031, 10024)] * 6:   # flicker: alternate high/low each frame
        _feed(p, wide=hi)
        assert not p._latest_violations
        _feed(p, wide=lo)
        assert not p._latest_violations


def test_spoof_outlier_below_floor_flagged():
    p = _mk(H)
    _warmup(p)
    _feed(p, wide=100)               # spoofed value far below the recent counter floor
    assert p._latest_violations and WIDE in p._latest_violations[0][0]


def test_reset_to_zero_not_flagged():
    p = _mk(H)
    _warmup(p)
    _feed(p, wide=0)                 # reset/wrap toward 0 (≤ DropFloor)
    assert not p._latest_violations


def test_spoof_emits_ex0014_02_incident():
    p = _mk(H)
    _warmup(p)
    closed = []
    p._write_incident = lambda inc: closed.append(inc)
    _feed(p, wide=100)               # violation
    _feed(p, wide=10032)             # clean (climb resumes) → closes incident
    assert closed, "a spoof violation should produce a closed incident"
    assert closed[0].cluster == "EX-0014.02" and WIDE in closed[0].sub_technique
    assert closed[0].mode == "SUNSAFE"
