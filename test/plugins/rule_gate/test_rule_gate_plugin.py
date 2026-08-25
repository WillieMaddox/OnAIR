# GSC-19165-1 OnAIR — rule_gate plugin tests
import os, io, contextlib, tempfile
import pytest

import sys
# onair package importable from fsw/ (conftest adds it); AIPlugin is a plain base.
from plugins.rule_gate.rule_gate_plugin import (Plugin, _to_num, _device_label,
                                                _incident_label)

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


def test_monstate_label_names_de0001_family():
    # AINOS3-73: DE-0001 (disable fault management) shares the LC-disable
    # footprint with EX-0011 / DE-0005 (all drive LC.CurrentLCState 1->3), so
    # R5's label names the whole fault-management/safe-mode family.
    from plugins.rule_gate.rule_gate_plugin import _monstate_label
    label = _monstate_label("R5:LC-monstate")
    assert "DE-0001" in label and "DE-0005" in label


def test_r5_catches_de0001_lc_disable():
    # DE-0001's on-board footprint is LC.CurrentLCState leaving its ACTIVE
    # baseline for DISABLED(3) — the same signal EX-0011/DE-0005 produce. R5
    # latches on it regardless of which SPARTA technique drove the disable.
    p = _mk(HM)
    _warmup_m(p)                         # LC baseline = ACTIVE(1)
    for _ in range(6):                   # DE-0001 SET_LC_STATE -> DISABLED(3)
        _feed_m(p, "3")
    assert any("LC-monstate" in r for r in p._latest_active)


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


# ── R14 ADCS mode-force (AINOS3-77) ──────────────────────────────────────────
# Mode codes (cmd.py): 0=PASSIVE 1=BDOT 2=SUNSAFE 3=INERTIAL.
# Note HRT already carries ADCS_GNC.Mode at a constant "2", so every R13 test
# above doubles as an implicit R14 no-fire check.

def _feed_mode(p, mode):
    _feed_rt(p, 0, 0, mode=str(mode))


def _warmup_mode(p, mode=2):
    for _ in range(31):
        _feed_mode(p, mode)


def test_r14_mode_force_latches():
    p = _mk(HRT)
    _warmup_mode(p, 2)                    # baseline SUNSAFE
    for _ in range(6):                    # forced SET_MODE INERTIAL, sustained
        _feed_mode(p, 3)
    assert "R14:adcs-mode" in p._latest_active


def test_r14_static_mode_never_fires():
    p = _mk(HRT)
    _warmup_mode(p, 2)
    for _ in range(40):                   # holding a mode is not a transition
        _feed_mode(p, 2)
    assert "R14:adcs-mode" not in p._latest_active


def test_r14_debounce_rejects_double_buffer_flicker():
    """The AINOS3-81 soak finding: OnAIR's double buffer oscillates old/new at
    every switch, and naive change-detection fired 22 times for 4 real
    transitions. A candidate that never persists `ModeDebounceFrames` frames
    must not be treated as a transition."""
    p = _mk(HRT)
    _warmup_mode(p, 2)
    for _ in range(20):
        _feed_mode(p, 3)                  # fresh buffer
        _feed_mode(p, 2)                  # stale buffer flickers back
    assert "R14:adcs-mode" not in p._latest_active
    assert p._mode_stable["ADCS_GNC.Mode"] == 2.0


def test_r14_rebaselines_to_the_new_mode():
    """Unlike R5/R13 (which latch until the field returns to baseline), a mode
    force is one bounded event: after the dwell the new mode is the new normal."""
    p = _mk(HRT)
    _warmup_mode(p, 2)
    for _ in range(6):
        _feed_mode(p, 3)
    assert "R14:adcs-mode" in p._latest_active
    assert p._mode_stable["ADCS_GNC.Mode"] == 3.0
    for _ in range(40):                   # keep holding INERTIAL
        _feed_mode(p, 3)
    assert "R14:adcs-mode" not in p._latest_active


def test_r14_fires_again_on_a_second_transition():
    p = _mk(HRT)
    _warmup_mode(p, 2)
    for _ in range(6):
        _feed_mode(p, 3)
    for _ in range(40):
        _feed_mode(p, 3)                  # quiesce
    for _ in range(6):
        _feed_mode(p, 0)                  # forced again: INERTIAL -> PASSIVE
    assert "R14:adcs-mode" in p._latest_active
    assert p._mode_last_transition == ("INERTIAL", "PASSIVE")


def test_r14_records_the_transition_for_the_operator_label():
    p = _mk(HRT)
    _warmup_mode(p, 2)
    for _ in range(6):
        _feed_mode(p, 3)
    assert p._mode_last_transition == ("SUNSAFE", "INERTIAL")


def test_r14_emits_de0005_incident():
    p = _mk(HRT)
    _warmup_mode(p, 2)
    closed = []
    p._write_incident = lambda inc: closed.append(inc)
    for _ in range(8):
        _feed_mode(p, 3)
    for _ in range(30):                   # dwell expires -> leaky decays -> closes
        _feed_mode(p, 3)
    assert closed, "a mode-force incident should open then close"
    assert closed[0].cluster == "DE-0005"
    assert "adcs-mode-force" in closed[0].sub_technique


def test_r14_yields_to_r5_when_both_fire():
    """DE-0005 subverts safe mode by disabling LC *and* forcing a mode. R5's
    EX-0011 is the family representative, so the incident keeps that label."""
    from plugins.rule_gate.rule_gate_plugin import _incident_label
    cluster, _ = _incident_label(["R14:adcs-mode", "R5:LC-monstate"])
    assert cluster == "EX-0011"
    cluster, sub = _incident_label(["R14:adcs-mode"])
    assert cluster == "DE-0005" and sub == "adcs-mode-force"


def test_r14_not_registered_when_mode_field_absent():
    p = _mk(H)                            # H has no ADCS_GNC.Mode
    assert p._modestate_idx == {}
    assert "R14:adcs-mode" not in p._rule_ids


def test_r14_ignores_init_sentinel():
    p = _mk(HRT)
    _warmup_mode(p, 2)
    for _ in range(10):
        _feed(p, HRT, {"IMU.DeviceEnabled": "1",
                       "CFE_EVS_HK.MessageSendCounter": "131",
                       "CFE_SB.MsgSendErrorCounter": "0",
                       "ADCS_HK.CommandErrorCount": "0",
                       "TO.usEnabledRoutes": "0", "TO.usConfigRoutes": "0",
                       "ADCS_GNC.Mode": "[0]"})
    assert "R14:adcs-mode" not in p._latest_active
    assert p._mode_stable["ADCS_GNC.Mode"] == 2.0


# ── R14 mode-flapping sub-rule (AINOS3-77 follow-on) ─────────────────────────
# A single switch is R14:adcs-mode. Repeated switching is a materially stronger
# signal: each one re-arms the IF's ~45 s post-switch blind window, so an
# attacker switching faster than that holds dynamics detection off indefinitely.

def _transition(p, to_mode, hold=8):
    """One confirmed transition: debounce frames to confirm, then settle."""
    for _ in range(hold):
        _feed_mode(p, to_mode)


def test_r14_flap_does_not_fire_on_a_single_transition():
    p = _mk(HRT)
    _warmup_mode(p, 2)
    _transition(p, 3)
    assert "R14:adcs-mode" in p._latest_active
    assert "R14:adcs-mode-flap" not in p._latest_active


def test_r14_flap_does_not_fire_below_the_threshold():
    p = _mk(HRT)
    _warmup_mode(p, 2)
    _transition(p, 3)
    _transition(p, 0)                     # 2 transitions — still under the min of 3
    assert "R14:adcs-mode-flap" not in p._latest_active


def test_r14_flap_fires_on_the_third_transition_in_window():
    p = _mk(HRT)
    _warmup_mode(p, 2)
    _transition(p, 3)
    _transition(p, 0)
    _transition(p, 1)                     # 3rd inside the window
    assert "R14:adcs-mode-flap" in p._latest_active
    assert p._mode_flap_count >= 3


def test_r14_flap_ignores_transitions_older_than_the_window():
    """Two transitions now and one from long ago is not flapping."""
    p = _mk(HRT)
    _warmup_mode(p, 2)
    _transition(p, 3)                     # transition 1
    for _ in range(p._mode_flap_window + 20):   # let it age out of the window
        _feed_mode(p, 3)
    _transition(p, 0)                     # transition 2 (1 is now stale)
    _transition(p, 1)                     # transition 3
    # Only 2 transitions are inside the window, so no flap.
    assert "R14:adcs-mode-flap" not in p._latest_active


def test_r14_flap_emits_a_distinct_incident_label():
    p = _mk(HRT)
    _warmup_mode(p, 2)
    closed = []
    p._write_incident = lambda inc: closed.append(inc)
    _transition(p, 3)
    _transition(p, 0)
    _transition(p, 1)
    for _ in range(40):                   # quiesce so the incident closes
        _feed_mode(p, 1)
    assert closed, "a flapping incident should open then close"
    assert closed[0].cluster == "DE-0005"
    assert "flapping" in closed[0].sub_technique


def test_r14_flap_label_distinguishes_from_single_force():
    from plugins.rule_gate.rule_gate_plugin import _incident_label
    c1, s1 = _incident_label(["R14:adcs-mode"])
    c2, s2 = _incident_label(["R14:adcs-mode-flap"])
    assert c1 == c2 == "DE-0005"
    assert s1 == "adcs-mode-force" and s2 == "adcs-mode-flapping"


def test_r14_flap_not_registered_without_the_mode_field():
    p = _mk(H)
    assert "R14:adcs-mode-flap" not in p._rule_ids


# ---------------------------------------------------------------------------
# R15 gps-time-divergence (AINOS3-95)
#
# The rule compares the MONOTONIC ENVELOPE of the GPS clock against that of the
# flight-software clock. Instantaneous differencing is unusable: measured on a
# clean stack the raw per-frame divergence has a 29 s spread because OnAIR's
# double buffer corrupts both operands (GPS steps backward, MET oscillates
# between two values ~4 s apart). Running maxima collapse that to 4.5 s.
# ---------------------------------------------------------------------------

HTD = ["IMU.DeviceEnabled", "CFE_EVS_HK.MessageSendCounter",
       "CFE_SB.MsgSendErrorCounter", "ADCS_HK.CommandErrorCount",
       "NOVATEL.Novatel_oem615.Weeks", "NOVATEL.Novatel_oem615.SecondsIntoWeek",
       "NOVATEL.Novatel_oem615.Fractions",
       "CFE_TIME.SecondsMET", "CFE_TIME.SubsecsMET",
       "CFE_TIME.SecondsSTCF", "CFE_TIME.SubsecsSTCF"]

_WEEK = 604800.0


def _feed_td(p, gps_s, met_s, weeks=341, stcf_s=0.0):
    """One frame. FSW clock = MET + STCF; a SET_TIME attack moves STCF, a SET_MET
    attack moves MET, a GPS spoof moves the GPS second-into-week."""
    _feed(p, HTD, {"IMU.DeviceEnabled": "1", "CFE_EVS_HK.MessageSendCounter": "131",
                   "CFE_SB.MsgSendErrorCounter": "0", "ADCS_HK.CommandErrorCount": "0",
                   "NOVATEL.Novatel_oem615.Weeks": str(weeks),
                   "NOVATEL.Novatel_oem615.SecondsIntoWeek": str(gps_s),
                   "NOVATEL.Novatel_oem615.Fractions": "0.0",
                   "CFE_TIME.SecondsMET": str(met_s), "CFE_TIME.SubsecsMET": "0",
                   "CFE_TIME.SecondsSTCF": str(stcf_s), "CFE_TIME.SubsecsSTCF": "0"})


def _warmup_td(p, n=31, gps0=150240.0, met0=21.0):
    """Nominal: both clocks advance together at ~1 s per frame."""
    for k in range(n):
        _feed_td(p, gps0 + k, met0 + k)
    return gps0 + n - 1, met0 + n - 1


def test_r15_registered_when_both_clocks_present():
    p = _mk(HTD)
    assert "R15:gps-time-divergence" in p._rule_ids


def test_r15_not_registered_without_the_gps_fields():
    p = _mk(HRT)                     # no NOVATEL / CFE_TIME columns
    assert "R15:gps-time-divergence" not in p._rule_ids


def test_r15_nominal_never_fires():
    """Both clocks advancing 1:1 is the nominal case and must stay silent."""
    p = _mk(HTD)
    g, m = _warmup_td(p)
    for k in range(1, 60):
        _feed_td(p, g + k, m + k)
    assert "R15:gps-time-divergence" not in p._latest_active


def test_r15_ignores_unacquired_receiver():
    """Weeks == 0 is 'no GPS fix yet', not a clock sitting at the GPS epoch.
    Treating it as real would make the envelope diverge by ~206 million s."""
    p = _mk(HTD)
    for _ in range(40):
        _feed_td(p, 0.0, 21.0, weeks=0)
    assert p._td_gmax is None
    assert "R15:gps-time-divergence" not in p._latest_active


def test_r15_detects_a_forward_time_jump():
    """A SET_TIME / spoof pushing GPS forward moves the GPS envelope off the
    baseline (EX-0014.01 / EX-0012.12)."""
    p = _mk(HTD)
    g, m = _warmup_td(p)
    for k in range(1, 6):
        _feed_td(p, g + k + 30.0, m + k)          # +30 s spoof
    assert "R15:gps-time-divergence" in p._latest_active


def test_r15_detects_a_live_set_time_via_stcf():
    """The path a real SET_TIME (0x1805 FC7) exercises, and the one the first
    version of this rule MISSED: cFE SET_TIME moves STCF, not the free-running
    MET. Confirmed live 2026-08-25 — STCF jumped 0 -> 199,999,092 while MET kept
    counting. The FSW clock is MET + STCF, so enveloping their sum catches it."""
    p = _mk(HTD)
    g, m = _warmup_td(p)                        # STCF = 0 throughout warmup
    for k in range(1, 6):
        _feed_td(p, g + k, m + k, stcf_s=30.0)  # SET_TIME shifts STCF by +30 s
    assert "R15:gps-time-divergence" in p._latest_active


def test_r15_detects_a_backward_time_jump():
    """Backward is detectable too, by a different mechanism: the GPS envelope
    FREEZES while the MET envelope keeps advancing, so the divergence shrinks."""
    p = _mk(HTD)
    g, m = _warmup_td(p)
    for k in range(1, 40):
        _feed_td(p, g - 60.0 + k, m + k)          # -60 s spoof, then resume
    assert "R15:gps-time-divergence" in p._latest_active


def test_r15_tolerates_double_buffer_flicker():
    """The measured pathology: GPS stepping BACKWARD one second and MET
    oscillating ~4 s between adjacent frames. Neither is an attack, and the
    running-max envelope must absorb both."""
    p = _mk(HTD)
    g, m = _warmup_td(p)
    for k in range(1, 60):
        back = -1.0 if k % 3 == 0 else 0.0        # GPS flicker backward
        stale = -4.0 if k % 2 == 0 else 0.0       # MET oscillation
        _feed_td(p, g + k + back, m + k + stale)
    assert "R15:gps-time-divergence" not in p._latest_active


def test_r15_small_jump_below_the_sawtooth_is_not_claimed():
    """Honest limit: a jump inside the ~4.5 s sawtooth floor is invisible. The
    test pins the limitation so it cannot be quietly overstated later."""
    p = _mk(HTD)
    g, m = _warmup_td(p)
    for k in range(1, 20):
        _feed_td(p, g + k + 5.0, m + k)           # +5 s, under the 9 s threshold
    assert "R15:gps-time-divergence" not in p._latest_active


def test_r15_emits_ex0014_01_incident():
    p = _mk(HTD)
    g, m = _warmup_td(p)
    for k in range(1, 6):
        _feed_td(p, g + k + 30.0, m + k)
    cluster, sub = _incident_label(["R15:gps-time-divergence"])
    assert cluster == "EX-0014.01"
    assert sub == "gps-met-divergence"


def test_r15_rejects_an_isolated_torn_read():
    """The nominal-soak false positive (2026-08-25): 2 of 3344 NOVATEL frames
    carried a torn read — Weeks 341 -> 43387, SecondsIntoWeek 150240 -> 731.
    A raw running max LATCHED that garbage and pinned R15 on for 2420 frames.
    The median pre-filter must reject an isolated spurious sample outright."""
    p = _mk(HTD)
    g, m = _warmup_td(p)
    for k in range(1, 60):
        if k == 20:                                  # single torn frame
            _feed_td(p, 731.0, m + k, weeks=43387)
        else:
            _feed_td(p, g + k, m + k)
    assert "R15:gps-time-divergence" not in p._latest_active


def test_r15_survives_frames_with_no_gps_fix():
    """Weeks == 0 frames must not crash the rule or corrupt the filter — the
    envelope block runs every frame, including ones with no usable clock."""
    p = _mk(HTD)
    g, m = _warmup_td(p)
    for k in range(1, 30):
        _feed_td(p, 0.0, m + k, weeks=0)             # no fix
    for k in range(30, 60):
        _feed_td(p, g + k, m + k)                    # fix returns, nominal
    assert "R15:gps-time-divergence" not in p._latest_active
