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


# R5 monitor-state: header includes LC.CurrentLCState (the default watched field)
HM = ["IMU.DeviceEnabled", "CFE_EVS_HK.MessageSendCounter",
      "CFE_SB.MsgSendErrorCounter", "ADCS_HK.CommandErrorCount", "LC.CurrentLCState"]


def _warmup_m(p, lc="1"):
    # 31 nominal frames establishing LC baseline = ACTIVE(1)
    for k in range(31):
        _feed(p, HM, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": str(100 + k),
                      "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0",
                      "LC.CurrentLCState": lc})


def _feed_m(p, lc):
    _feed(p, HM, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": "131",
                  "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0",
                  "LC.CurrentLCState": lc})


def test_monstate_label_maps_ex0011():
    from plugins.rule_gate.rule_gate_plugin import _monstate_label
    assert "EX-0011" in _monstate_label("R5:LC-monstate")


def test_r5_monitor_state_disable_latches_and_clears():
    p = _mk(HM)
    _warmup_m(p)                         # LC baseline = ACTIVE(1)
    for _ in range(6):                   # LC -> DISABLED(3), sustained
        _feed_m(p, "3")
    assert any("LC-monstate" in r for r in p._latest_active)
    for _ in range(10):                  # restored to ACTIVE -> leaky decays -> clears
        _feed_m(p, "1")
    assert p._latest_active == []


def test_r5_no_alert_when_lc_never_received():
    # LC.CurrentLCState stays the [0] sentinel -> no baseline -> R5 never fires
    p = _mk(HM)
    for _ in range(40):
        _feed(p, HM, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": "131",
                      "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0"})
    assert not any("R5" in r for r in p._latest_active)


def test_r5_emits_ex0011_incident():
    p = _mk(HM)
    _warmup_m(p)
    for _ in range(10):
        _feed_m(p, "3")
    closed = []
    p._write_incident = lambda inc: closed.append(inc)
    for _ in range(10):                  # restore -> alert clears -> incident closes
        _feed_m(p, "1")
    assert closed, "an LC-disable incident should close on restore"
    assert closed[0].cluster == "EX-0011" and "LC" in closed[0].sub_technique


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


# R6 sb-command: header includes CFE_SB.CommandCounter
HSB = ["IMU.DeviceEnabled", "CFE_EVS_HK.MessageSendCounter",
       "CFE_SB.MsgSendErrorCounter", "ADCS_HK.CommandErrorCount",
       "CFE_SB.CommandCounter", "ADCS_GNC.Mode"]


def _feed_sb(p, cc, mode="2"):
    _feed(p, HSB, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": "131",
                   "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0",
                   "CFE_SB.CommandCounter": str(cc), "ADCS_GNC.Mode": mode})


def _warmup_sb(p, cc=1):
    for _ in range(31):
        _feed_sb(p, cc)                       # CFE_SB.CommandCounter static at baseline


def test_r6_sb_command_latches():
    p = _mk(HSB)
    _warmup_sb(p, cc=1)                        # baseline CommandCounter = 1
    _feed_sb(p, 2)                            # a CFE_SB command: 1->2 new high -> dwell
    for _ in range(4):                        # counter static; dwell keeps firing -> latch
        _feed_sb(p, 2)
    assert "R6:sb-command" in p._latest_active


def test_r6_static_never_fires():
    p = _mk(HSB)
    _warmup_sb(p, cc=1)
    for _ in range(20):                       # no command -> counter static
        _feed_sb(p, 1)
    assert "R6:sb-command" not in p._latest_active


def test_r6_double_buffer_flicker_single_command():
    # one command flickering 2<->1 (stale buffer): running max ignores the stale 1s
    p = _mk(HSB)
    _warmup_sb(p, cc=1)
    for cc in [2, 1, 2, 1, 2, 1]:
        _feed_sb(p, cc)
    assert "R6:sb-command" in p._latest_active


def test_r6_emits_ex0012_02_incident():
    p = _mk(HSB)
    _warmup_sb(p, cc=1)
    closed = []
    p._write_incident = lambda inc: closed.append(inc)
    _feed_sb(p, 2)                            # command -> dwell -> latch
    for _ in range(28):                       # counter static; dwell ends -> decays -> clears
        _feed_sb(p, 2)
    assert closed, "an sb-command incident should open then close"
    assert closed[0].cluster == "EX-0012.02" and "sb-command" in closed[0].sub_technique


# R7 evs-command: header includes CFE_EVS_HK.CommandCounter (static in nominal)
HEV = ["IMU.DeviceEnabled", "CFE_EVS_HK.MessageSendCounter",
       "CFE_SB.MsgSendErrorCounter", "ADCS_HK.CommandErrorCount",
       "CFE_EVS_HK.CommandCounter", "ADCS_GNC.Mode"]


def _feed_ev(p, cc, mode="2"):
    _feed(p, HEV, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": "131",
                   "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0",
                   "CFE_EVS_HK.CommandCounter": str(cc), "ADCS_GNC.Mode": mode})


def _warmup_ev(p, cc=1):
    for _ in range(31):
        _feed_ev(p, cc)


def test_r7_evs_command_latches():
    p = _mk(HEV)
    _warmup_ev(p, cc=1)                          # baseline CFE_EVS CommandCounter = 1
    _feed_ev(p, 2)                              # a CFE_EVS command: 1->2 new high -> dwell
    for _ in range(4):
        _feed_ev(p, 2)
    assert "R7:evs-command" in p._latest_active


def test_r7_static_never_fires():
    p = _mk(HEV)
    _warmup_ev(p, cc=1)
    for _ in range(20):                          # no command -> counter static
        _feed_ev(p, 1)
    assert "R7:evs-command" not in p._latest_active


def test_r7_emits_de0002_03_incident():
    p = _mk(HEV)
    _warmup_ev(p, cc=1)
    closed = []
    p._write_incident = lambda inc: closed.append(inc)
    _feed_ev(p, 2)                              # command -> dwell -> latch
    for _ in range(28):                          # static; dwell ends -> decays -> clears
        _feed_ev(p, 2)
    assert closed, "an evs-command incident should open then close"
    assert closed[0].cluster == "DE-0002.03" and "evs-command" in closed[0].sub_technique
