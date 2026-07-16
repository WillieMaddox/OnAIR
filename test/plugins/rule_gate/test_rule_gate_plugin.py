# GSC-19165-1 OnAIR — rule_gate plugin tests
import os, io, contextlib, tempfile
import pytest

import sys
# onair package importable from fsw/ (conftest adds it); AIPlugin is a plain base.
from plugins.rule_gate.rule_gate_plugin import Plugin, _to_num, _device_label

# A real ini that disables file output, so construction never touches the
# runtime-relative ../../../../data dir (unwritable from the pytest cwd). The
# tests only inspect in-memory state / monkeypatch _write_incident.
_TEST_INI = os.path.join(tempfile.gettempdir(), "rule_gate_test.ini")
with open(_TEST_INI, "w") as _f:
    _f.write("[RULE_GATE]\nWriteSideFile=false\nWriteIncidentFile=false\n")


def _mk(headers):
    os.environ["ONAIR_INI_FILE"] = _TEST_INI
    with contextlib.redirect_stdout(io.StringIO()):
        p = Plugin("rule_gate", headers)
    p._side_file_path = None
    p._incident_file_path = None
    return p


def _feed(p, headers, row):
    lld = [row.get(h, "[0]") for h in headers]
    with contextlib.redirect_stdout(io.StringIO()):
        p.update(lld, {})


H = ["IMU.DeviceEnabled", "CFE_EVS_HK.MessageSendCounter",
     "CFE_SB.MsgSendErrorCounter", "ADCS_HK.CommandErrorCount"]


def _warmup(p, evs0=100):
    # 31 nominal frames: IMU enabled, EVS creeping slowly, no SB/cmd errors
    for k in range(31):
        _feed(p, H, {"IMU.DeviceEnabled": "1",
                     "CFE_EVS_HK.MessageSendCounter": str(evs0 + k),
                     "CFE_SB.MsgSendErrorCounter": "0",
                     "ADCS_HK.CommandErrorCount": "0"})


def test_to_num_handles_init_sentinel_and_text():
    assert _to_num("[0]") is None
    assert _to_num("5") == 5.0
    assert _to_num(None) is None


def test_no_alert_during_warmup_and_when_nominal():
    p = _mk(H)
    _warmup(p)
    assert p._latest_active == []


def test_r1_device_disable_latches_after_baseline():
    p = _mk(H)
    _warmup(p)                       # baseline IMU.DeviceEnabled = 1
    for _ in range(6):               # sustained disable → leaky climbs to alert
        _feed(p, H, {"IMU.DeviceEnabled": "0", "CFE_EVS_HK.MessageSendCounter": "131",
                     "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0"})
    assert any("IMU-disabled" in r for r in p._latest_active)
    # re-enable → leaky decays → clears
    for _ in range(10):
        _feed(p, H, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": "131",
                     "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0"})
    assert p._latest_active == []


def test_r2_evs_flood_fires():
    p = _mk(H)
    _warmup(p, evs0=100)
    evs = 131
    for _ in range(6):               # +50 events/frame >> 15 threshold
        evs += 50
        _feed(p, H, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": str(evs),
                     "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0"})
    assert "R2:evs" in p._latest_active


def test_leaky_tolerates_every_other_frame_flicker():
    # R2 flood signal that fires only every OTHER frame (double-buffer pattern)
    p = _mk(H)
    _warmup(p, evs0=100)
    evs = 131
    for k in range(12):
        evs += 50 if k % 2 == 0 else 0     # spike, flat, spike, flat, ...
        _feed(p, H, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": str(evs),
                     "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0"})
    assert "R2:evs" in p._latest_active, "leaky integrator should latch despite flicker"


def test_device_label_maps_technique():
    assert "EX-0002" in _device_label("R1:NOVATEL_HK-disabled")
    assert "EX-0014.03" in _device_label("R1:IMU-disabled")
    assert "DE-0002.03" in _device_label("R1:EPS-disabled")


def test_incident_emitted_for_device_disable():
    p = _mk(H)
    _warmup(p)
    # sustained disable long enough to open + (on re-enable) close an incident
    for _ in range(10):
        _feed(p, H, {"IMU.DeviceEnabled": "0", "CFE_EVS_HK.MessageSendCounter": "131",
                     "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0"})
    closed = []
    orig = p._write_incident
    p._write_incident = lambda inc: closed.append(inc)
    for _ in range(10):   # re-enable -> alert clears -> incident closes
        _feed(p, H, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": "131",
                     "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0"})
    assert closed, "a device-disable incident should close on re-enable"
    inc = closed[0]
    assert inc.cluster == "EX-0014.03" and "IMU" in inc.sub_technique
