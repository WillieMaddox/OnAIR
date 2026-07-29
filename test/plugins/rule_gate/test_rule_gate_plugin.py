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


# R8 es-command: header includes CFE_ES.CommandCounter (static in nominal). A
# CFE_ES command (SET_MAX_PR_COUNT etc.) modifies a C&DH on-board value the IF
# and the consistency/staleness gates all miss — EX-0012.10.
HES = ["IMU.DeviceEnabled", "CFE_EVS_HK.MessageSendCounter",
       "CFE_SB.MsgSendErrorCounter", "ADCS_HK.CommandErrorCount",
       "CFE_ES.CommandCounter", "ADCS_GNC.Mode"]


def _feed_es(p, cc, mode="2"):
    _feed(p, HES, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": "131",
                   "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0",
                   "CFE_ES.CommandCounter": str(cc), "ADCS_GNC.Mode": mode})


def _warmup_es(p, cc=0):
    for _ in range(31):
        _feed_es(p, cc)


def test_r8_es_command_latches():
    p = _mk(HES)
    _warmup_es(p, cc=0)                          # baseline CFE_ES CommandCounter = 0
    _feed_es(p, 4)                              # a CFE_ES command burst: 0->4 new high
    for _ in range(4):
        _feed_es(p, 4)
    assert "R8:es-command" in p._latest_active


def test_r8_static_never_fires():
    p = _mk(HES)
    _warmup_es(p, cc=0)
    for _ in range(20):                          # no command -> counter static
        _feed_es(p, 0)
    assert "R8:es-command" not in p._latest_active


def test_r8_emits_ex0012_10_incident():
    p = _mk(HES)
    _warmup_es(p, cc=0)
    closed = []
    p._write_incident = lambda inc: closed.append(inc)
    _feed_es(p, 4)                              # command -> dwell -> latch
    for _ in range(28):                          # static; dwell ends -> decays -> clears
        _feed_es(p, 4)
    assert closed, "an es-command incident should open then close"
    assert closed[0].cluster == "EX-0012.10" and "es-command" in closed[0].sub_technique


# R9 tbl-command: header includes CFE_TBL.CommandCounter (static in nominal). A
# CFE_TBL LOAD/ACTIVATE is table-backdoor persistence (PER-0001) the IF and the
# consistency/staleness gates all miss.
HTB = ["IMU.DeviceEnabled", "CFE_EVS_HK.MessageSendCounter",
       "CFE_SB.MsgSendErrorCounter", "ADCS_HK.CommandErrorCount",
       "CFE_TBL.CommandCounter", "ADCS_GNC.Mode"]


def _feed_tb(p, cc, mode="2"):
    _feed(p, HTB, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": "131",
                   "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0",
                   "CFE_TBL.CommandCounter": str(cc), "ADCS_GNC.Mode": mode})


def _warmup_tb(p, cc=0):
    for _ in range(31):
        _feed_tb(p, cc)


def test_r9_tbl_command_latches():
    p = _mk(HTB)
    _warmup_tb(p, cc=0)                          # baseline CFE_TBL CommandCounter = 0
    _feed_tb(p, 1)                              # a CFE_TBL command: 0->1 new high
    for _ in range(4):
        _feed_tb(p, 1)
    assert "R9:tbl-command" in p._latest_active


def test_r9_survives_evidence_hiding_reset():
    p = _mk(HTB)
    _warmup_tb(p, cc=0)
    _feed_tb(p, 2)                              # commands land: new high 0->2
    _feed_tb(p, 0)                              # attacker CFE_TBL_RESET zeroes it
    for _ in range(3):
        _feed_tb(p, 0)
    # the running-max latched the new high before the reset -> still alerting via dwell
    assert "R9:tbl-command" in p._latest_active


def test_r9_static_never_fires():
    p = _mk(HTB)
    _warmup_tb(p, cc=0)
    for _ in range(20):
        _feed_tb(p, 0)
    assert "R9:tbl-command" not in p._latest_active


def test_r9_emits_per0001_incident():
    p = _mk(HTB)
    _warmup_tb(p, cc=0)
    closed = []
    p._write_incident = lambda inc: closed.append(inc)
    _feed_tb(p, 1)
    for _ in range(28):
        _feed_tb(p, 1)
    assert closed, "a tbl-command incident should open then close"
    assert closed[0].cluster == "PER-0001" and "tbl-command" in closed[0].sub_technique


# R10 bus-sweep: LM-0002 sweeps every MID, tripping several static-in-nominal command
# counters at once. The header carries all four (CFE_SB/EVS/ES/TBL) so >=3 firing
# together crosses BusSweepMinRules and labels LM-0002 instead of collapsing to R2.
HSW = ["IMU.DeviceEnabled", "CFE_EVS_HK.MessageSendCounter",
       "CFE_SB.MsgSendErrorCounter", "ADCS_HK.CommandErrorCount",
       "CFE_SB.CommandCounter", "CFE_EVS_HK.CommandCounter",
       "CFE_ES.CommandCounter", "CFE_TBL.CommandCounter", "ADCS_GNC.Mode"]


def _feed_sw(p, sb, evs, es, tbl, mode="2"):
    _feed(p, HSW, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": "131",
                   "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0",
                   "CFE_SB.CommandCounter": str(sb), "CFE_EVS_HK.CommandCounter": str(evs),
                   "CFE_ES.CommandCounter": str(es), "CFE_TBL.CommandCounter": str(tbl),
                   "ADCS_GNC.Mode": mode})


def _warmup_sw(p):
    for _ in range(31):
        _feed_sw(p, 0, 0, 0, 0)


def test_r10_bus_sweep_latches():
    p = _mk(HSW)
    _warmup_sw(p)
    _feed_sw(p, 1, 1, 1, 1)                      # 4 command counters tick together = sweep
    for _ in range(4):
        _feed_sw(p, 1, 1, 1, 1)
    assert "R10:bus-sweep" in p._latest_active


def test_r10_single_command_does_not_fire():
    p = _mk(HSW)
    _warmup_sw(p)
    _feed_sw(p, 1, 0, 0, 0)                      # only CFE_SB moved (a single technique)
    for _ in range(4):
        _feed_sw(p, 1, 0, 0, 0)
    assert "R6:sb-command" in p._latest_active   # the single command rule still fires
    assert "R10:bus-sweep" not in p._latest_active


def test_r10_emits_lm0002_incident_over_command_rules():
    p = _mk(HSW)
    _warmup_sw(p)
    closed = []
    p._write_incident = lambda inc: closed.append(inc)
    _feed_sw(p, 1, 1, 1, 1)                      # sweep -> R6+R7+R8+R9+R10
    for _ in range(28):
        _feed_sw(p, 1, 1, 1, 1)
    assert closed, "a bus-sweep incident should open then close"
    # R10 outranks the individual command rules -> labeled LM-0002, not a single rule
    assert closed[0].cluster == "LM-0002" and "bus-sweep" in closed[0].sub_technique


# R11 fm-command: header includes FM.CommandCounter (static in nominal — validated
# live 2026-07-29 static at 0/1). A burst of FM file ops is the wiper (EX-0010.02
# DELETE_ALL) / ransomware (EX-0010.01 COPY->.enc + DELETE) footprint the dynamics-IF
# and the consistency/staleness gates all miss.
HFM = ["IMU.DeviceEnabled", "CFE_EVS_HK.MessageSendCounter",
       "CFE_SB.MsgSendErrorCounter", "ADCS_HK.CommandErrorCount",
       "FM.CommandCounter", "ADCS_GNC.Mode"]


def _feed_fm(p, cc, mode="2"):
    _feed(p, HFM, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": "131",
                   "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0",
                   "FM.CommandCounter": str(cc), "ADCS_GNC.Mode": mode})


def _warmup_fm(p, cc=0):
    for _ in range(31):
        _feed_fm(p, cc)


def test_r11_fm_command_latches():
    p = _mk(HFM)
    _warmup_fm(p, cc=1)                          # baseline FM CommandCounter = 1 (post-NOOP)
    _feed_fm(p, 24)                             # a wiper burst: 1->24 new high
    for _ in range(4):
        _feed_fm(p, 24)
    assert "R11:fm-command" in p._latest_active


def test_r11_static_never_fires():
    p = _mk(HFM)
    _warmup_fm(p, cc=0)
    for _ in range(20):                          # no FM command -> counter static
        _feed_fm(p, 0)
    assert "R11:fm-command" not in p._latest_active


def test_r11_double_buffer_flicker_single_burst():
    # The OnAIR double buffer flickers the fresh value with the stale one; the running
    # max must ignore the flicker back to the pre-burst value and keep the alert up.
    p = _mk(HFM)
    _warmup_fm(p, cc=1)
    for _ in range(6):
        _feed_fm(p, 71)                         # fresh buffer: ransomware peak
        _feed_fm(p, 1)                          # stale buffer flickers back to baseline
    assert "R11:fm-command" in p._latest_active


def test_r11_emits_ex0010_incident():
    p = _mk(HFM)
    _warmup_fm(p, cc=0)
    closed = []
    p._write_incident = lambda inc: closed.append(inc)
    _feed_fm(p, 23)                             # file-op burst -> dwell -> latch
    for _ in range(28):                          # static; dwell ends -> decays -> clears
        _feed_fm(p, 23)
    assert closed, "an fm-command incident should open then close"
    assert closed[0].cluster == "EX-0010" and "fm-command" in closed[0].sub_technique


# R12 to-command: header includes TO.usCmdCnt (the full TO app's command counter,
# static in nominal — validated live 2026-07-29 static at 0). A command to the full
# TO app (TO_ENABLE_OUTPUT redirects the downlink) is the EXF-0003.02 downlink-exfil
# signal the dynamics-IF and the other gates miss.
HTO = ["IMU.DeviceEnabled", "CFE_EVS_HK.MessageSendCounter",
       "CFE_SB.MsgSendErrorCounter", "ADCS_HK.CommandErrorCount",
       "TO.usCmdCnt", "ADCS_GNC.Mode"]


def _feed_to(p, cc, mode="2"):
    _feed(p, HTO, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": "131",
                   "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0",
                   "TO.usCmdCnt": str(cc), "ADCS_GNC.Mode": mode})


def _warmup_to(p, cc=0):
    for _ in range(31):
        _feed_to(p, cc)


def test_r12_to_command_latches():
    p = _mk(HTO)
    _warmup_to(p, cc=0)                          # baseline TO.usCmdCnt = 0 (nominal)
    _feed_to(p, 1)                              # a TO command: 0->1 new high
    for _ in range(4):
        _feed_to(p, 1)
    assert "R12:to-command" in p._latest_active


def test_r12_static_never_fires():
    p = _mk(HTO)
    _warmup_to(p, cc=0)
    for _ in range(20):                          # no TO command -> counter static
        _feed_to(p, 0)
    assert "R12:to-command" not in p._latest_active


def test_r12_emits_exf0003_02_incident():
    p = _mk(HTO)
    _warmup_to(p, cc=0)
    closed = []
    p._write_incident = lambda inc: closed.append(inc)
    _feed_to(p, 1)                              # ENABLE_OUTPUT redirect -> dwell -> latch
    for _ in range(28):
        _feed_to(p, 1)
    assert closed, "a to-command incident should open then close"
    assert closed[0].cluster == "EXF-0003.02" and "to-command" in closed[0].sub_technique


# R13 to-route: header includes the full TO app's downlink route masks. A change from
# the nominal baseline means the downlink was reconfigured (route enabled/disabled) —
# the specific EXF-0003.02 exfil corroborator of R12.
HRT = ["IMU.DeviceEnabled", "CFE_EVS_HK.MessageSendCounter",
       "CFE_SB.MsgSendErrorCounter", "ADCS_HK.CommandErrorCount",
       "TO.usEnabledRoutes", "TO.usConfigRoutes", "ADCS_GNC.Mode"]


def _feed_rt(p, enabled, config, mode="2"):
    _feed(p, HRT, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": "131",
                   "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0",
                   "TO.usEnabledRoutes": str(enabled), "TO.usConfigRoutes": str(config),
                   "ADCS_GNC.Mode": mode})


def _warmup_rt(p, enabled=0, config=0):
    for _ in range(31):
        _feed_rt(p, enabled, config)


def test_r13_route_change_latches():
    p = _mk(HRT)
    _warmup_rt(p, enabled=0, config=0)          # baseline: downlink routes off
    _feed_rt(p, 1, 1)                           # ENABLE_OUTPUT turns a route on: 0->1
    for _ in range(4):
        _feed_rt(p, 1, 1)
    assert "R13:to-route" in p._latest_active


def test_r13_static_never_fires():
    p = _mk(HRT)
    _warmup_rt(p, enabled=1, config=1)          # baseline: downlink already enabled
    for _ in range(20):                          # unchanged -> no fire
        _feed_rt(p, 1, 1)
    assert "R13:to-route" not in p._latest_active


def test_r13_tolerates_double_buffer_flicker():
    # usEnabledRoutes flickers fresh(1) / stale(0) after the change; the leaky
    # integrator must keep the alert up despite the every-other-frame flicker.
    p = _mk(HRT)
    _warmup_rt(p, enabled=0, config=0)
    for _ in range(6):
        _feed_rt(p, 1, 1)                       # fresh buffer: route enabled
        _feed_rt(p, 0, 0)                       # stale buffer flickers to baseline
    assert "R13:to-route" in p._latest_active


def test_r13_emits_exf0003_02_incident():
    p = _mk(HRT)
    _warmup_rt(p, enabled=0, config=0)
    closed = []
    p._write_incident = lambda inc: closed.append(inc)
    _feed_rt(p, 1, 1)
    for _ in range(6):
        _feed_rt(p, 1, 1)
    for _ in range(28):                          # route returns to baseline -> clears
        _feed_rt(p, 0, 0)
    assert closed, "a to-route incident should open then close"
    assert closed[0].cluster == "EXF-0003.02" and "to-route" in closed[0].sub_technique
