# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
sbn_adapter_blended — AINOS3-126

A drop-in replacement for `sbn_adapter` that reconstructs ONE coherent telemetry
frame from OnAIR's double buffer, and can emit it alongside (or instead of) the
raw interleaved stream.

WHY
---
`sbn_adapter.DataSource` keeps two buffers. The listener thread writes each
arriving message into the *write* buffer, `currentData[(read_index + 1) % 2]`,
and every frame `get_next()` flips `read_index` and returns the other one. The
two are never reconciled — the upstream comment says so outright:

    "The double buffer does not clear between switching. If fresh data doesn't
     come in, stale data is returned (delayed by 1 frame)"

So each buffer is an independent, partially-stale snapshot, and every CSV this
project has recorded is two interleaved sub-streams. `AINOS3-125` measured the
cost: 49 of 454 non-constant columns alternate on >50 % of steady-state rows,
and lag-1 deltas — what every model consumes — carry 4-6x the noise of
same-buffer deltas.

THE TRANSFORM
-------------
Exactly the one `training/deinterleave_csv.py` validated offline. A field is
news only when it changed against **its own buffer's** previous frame;
everything else carries forward:

    for each frame, belonging to buffer b:
        for each field where frame[field] != ref[b][field]:
            blended[field] = frame[field]       # a genuine update from b
        ref[b] = frame                          # before the next frame arrives
        emit blended                            # carries every other field forward

⚠ A naive single "latest value" dict is WRONG. Buffer B's copy of a field is
usually a STALE value it has been holding since the last message that happened
to land in B; letting it overwrite A's fresh update reproduces the flicker
exactly.

⚠ Parity subsampling is NOT an alternative either: on discrete baseline fields
55.6 % of the novel values an attack produces appear in ONE parity only, so
keeping one sub-stream discards over half the footprint evidence.

MODES  (`[SBN_ADAPTER] BlendMode`)
----------------------------------
  tap     (default, this sprint)  `get_next()` returns the RAW frame, exactly as
          the parent adapter does — the read path is untouched, so `csv_output`
          keeps writing the same `csv_out_*.csv` it always has and every
          deployed detector sees exactly what it saw before. This adapter
          side-writes the blended stream to its own `csv_blended_*.csv`.
          Both files exist, one row each per frame, so
          `deinterleave_csv.py(raw) == blended` is checkable per run.
          This is the AINOS3-126 AC2/AC2b verification mode.

  inline  (after AC2 is verified and AC4's refits land)  `get_next()` returns the
          BLENDED frame, so `csv_output` writes the blended stream and the raw
          interleaved stream is NOT recorded at all. No side file. This is the
          "we no longer need to save the alternating streams" end state, and the
          switch is a one-line ini change in both directions (AINOS3-37).
          ⚠ It also changes what the DETECTORS read, which is why it is gated on
          the IF and classifier being refitted on blended data — deploying it
          alone leaves both models scoring inputs whose distribution they were
          never fitted to.

  off     No blending, no side file. Byte-for-byte the parent adapter.

EQUIVALENCE, BY CONSTRUCTION
----------------------------
The blend runs over the *stringified* values — the exact domain
`deinterleave_csv.py` works in, because that script reads them back out of the
CSV. `_stringify` here is `csv_output_plugin._stringify`, and
`test_sbn_adapter_blended.py` pins the two implementations together. In `inline`
mode the blended OBJECTS are carried alongside the strings and returned to the
pipeline, so downstream value types are unchanged; the update decision is made
on the strings either way, so both representations stay in lock-step.

⚠ Buffer LABELS differ from the offline script and that is fine. The offline
script calls CSV row 0 "buffer 0"; here row 0 comes from the parent's
`double_buffer_read_index == 1`. The blend is symmetric under swapping the two
slots, so the output is identical — see `test_parity_label_symmetry`.
"""

from __future__ import annotations

import configparser
import csv
import hashlib
import json
import os
import socket
import threading
import time
from datetime import datetime, timedelta, timezone

import onair.data_handling.sbn_adapter as sbn_adapter


ADAPTER_VERSION = "sbn_adapter_blended@1.2"

# ⚠ A REAL per-row timestamp, stamped when the frame is read.
#
# Until now nothing in the recorded stream said when a row happened, so
# `loader._synthesize_row_times` reconstructed it as
# `file_start + row_idx / rate`, with `rate` inferred by
# `loader._compute_step_rates` from the gap to the NEXT FILE's start. That gap
# includes the ~2 min `make stop` + `launch-quiet` between runs, so the rate is
# systematically under-estimated and the error accumulates down the file:
# measured across the 113 `rebuild_2026-09-10` runs, the median row-time drift
# at end-of-file is 57 s and the worst is 188 s, against attack windows of a
# few minutes. That is a labelling error, not a cosmetic one.
#
# ⚠ The clocks already in the stream cannot fix it. `CFE_TIME.SecondsMET` is
# the best of them and it ticks once per 4.000 s — 24.9 frames — so it localises
# a row to +/-2 s at best; every other monotonic counter (SCH.SlotsProcessed,
# SCH.TablePass, LC.MonitoredMsg) advances on the same 4.0 % of frames, and GPS
# time advances on 16 % with torn reads. Interpolating between MET ticks is only
# sound on BLENDED data (raw MET steps backwards on 48.2 % of rows), and even
# then it is an estimate.
#
# So the frame carries its own arrival time, in the same ISO-8601 UTC form the
# attack manifests use for `start_utc`/`end_utc` — a direct comparison, with no
# rate inference and no interpolation anywhere in the chain.
TIMESTAMP_COLUMN = "OnAIR.FrameRecvUTC"

# ⚠ SIMULATION time, which is NOT the frame's arrival time and not MET.
#
# The sim starts from a fixed epoch configured in `cfg/InOut/Inp_Sim.txt`
# (`10 20 2025 / 17 43 20.00` by default), and every run to date has launched
# from that same instant — so the corpus samples exactly ONE orbital phase and
# has no eclipse/sunlit variation at all. Recording sim time makes that
# variation legible the moment the epoch is changed, and lets a run be placed
# in its orbit without re-deriving anything.
#
# It is NOT already available in a usable form:
#   * CFE_TIME.SecondsSTCF is 0, so cFS carries MET only — no absolute epoch.
#   * The NOVATEL GPS sim does carry it, but the week number is the standard
#     10-bit field, so a naive decode of week 341 gives 1986-07-21 instead of
#     2025-10-20. ⚠ The TIME OF DAY is correct either way, which makes the bug
#     quietly survivable and is exactly why it belongs in one place.
#   * GPS arrives on ~16 % of frames and suffers torn reads (measured: 2
#     backwards steps per 8k frames), so the raw fields are not directly usable.
SIM_TIME_COLUMN = "OnAIR.SimTimeUTC"
_GPS_WEEKS = "NOVATEL.Novatel_oem615.Weeks"
_GPS_SOW = "NOVATEL.Novatel_oem615.SecondsIntoWeek"
_LEAP_SECONDS = "CFE_TIME.LeapSeconds"
_GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
_GPS_ROLLOVER_WEEKS = 1024
_SECONDS_PER_WEEK = 604800


def read_sim_epoch(path="cfg/InOut/Inp_Sim.txt"):
    """The sim's configured start instant, for resolving the GPS week rollover.

    Read from 42's own input file rather than hard-coded, so changing the epoch
    to test a different orbital phase needs no code change — which is the whole
    point of recording sim time.

    Returns None if the file is unreadable; the caller then leaves sim time
    blank rather than emitting a date that is wrong by a multiple of ~19.6 years.
    """
    for base in ("", "../", "../../", "../../../", "../../../../"):
        p = os.path.join(base, path)
        if not os.path.exists(p):
            continue
        try:
            date_parts = time_parts = None
            with open(p, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if "!" not in line:
                        continue
                    val, _, comment = line.partition("!")
                    c = comment.lower()
                    if "date" in c and "utc" in c:
                        date_parts = [int(float(x)) for x in val.split()[:3]]
                    elif "time" in c and "utc" in c:
                        time_parts = [float(x) for x in val.split()[:3]]
            if date_parts and time_parts:
                mo, dy, yr = date_parts
                hh, mm, ss = time_parts
                return datetime(yr, mo, dy, int(hh), int(mm), int(ss),
                                tzinfo=timezone.utc)
        except (ValueError, OSError):
            return None
    return None


class SimClock:
    """GPS week/second pairs -> a monotonic absolute UTC string.

    Stateful on purpose. The raw fields alternate between buffers and suffer
    torn reads, so a stateless per-frame decode would flicker. Holding the last
    good value and refusing to go backwards means the value written INTO the
    frame is already clean — which is what lets the raw file and the blended
    file carry the identical value and keeps `deinterleave_csv(raw) == blend`
    true for this column, exactly as for the arrival timestamp.
    """

    def __init__(self, epoch=None, max_step_s=3600.0):
        self.epoch = epoch
        self.max_step_s = max_step_s
        self._rollovers = None     # resolved once, from the first good fix
        self._last = None          # last accepted datetime
        self.rejected = 0
        self.accepted = 0

    def _decode(self, weeks, sow, rollovers):
        return _GPS_EPOCH + timedelta(
            seconds=(weeks + rollovers * _GPS_ROLLOVER_WEEKS) * _SECONDS_PER_WEEK + sow)

    def update(self, weeks_raw, sow_raw, leap_raw):
        """→ ISO-8601 UTC string, or "" before the first good fix."""
        try:
            weeks = int(float(weeks_raw))
            sow = float(sow_raw)
            leap = float(leap_raw) if leap_raw not in (None, "", "[0]") else 0.0
        except (TypeError, ValueError):
            return self._iso()
        if weeks < 0 or not (0.0 <= sow < _SECONDS_PER_WEEK):
            self.rejected += 1
            return self._iso()

        if self._rollovers is None:
            if self.epoch is None:
                return ""
            # ⚠ Pick the 1024-week era whose decode lands nearest the CONFIGURED
            # epoch. Without this, week 341 decodes to 1986 rather than 2025.
            self._rollovers = min(
                range(0, 8),
                key=lambda n: abs((self._decode(weeks, sow, n) - self.epoch).total_seconds()))

        t = self._decode(weeks, sow, self._rollovers) - timedelta(seconds=leap)

        # Torn NOVATEL reads step backwards or jump absurdly; neither is real
        # sim time, so hold the last good value instead of recording a lie.
        if self._last is not None:
            delta = (t - self._last).total_seconds()
            if delta < 0 or delta > self.max_step_s:
                self.rejected += 1
                return self._iso()
        self._last = t
        self.accepted += 1
        return self._iso()

    def _iso(self):
        return self._last.isoformat() if self._last is not None else ""

# OnAIR's "this MID has never been received in this buffer" marker, as it reads
# in the CSV. Kept as the string because that is what the blend compares.
SENTINEL = "[0]"

# ⚠ The sentinel is WRITTEN, never blanked, and never ADOPTED over good
# telemetry. Both properties are required (AINOS3-125 AC2 / AINOS3-126 AC3):
#
#   * `[0]` must never OVERWRITE good telemetry. Buffer B's un-received copy of
#     a field is not news; adopting it would undo A's real value.
#   * a field NO buffer has ever received must still READ `[0]`, not "". Four
#     tools key on the literal sentinel and two fail SILENTLY on a blank:
#       - build_corpus_manifest.py:177  "" scores 1 (uncontrolled) not 0 (no
#         data), deflating the AINOS3-100 INERTIAL capture gate
#       - analyze_inertial_fp.py:195    "" lands in the UNCONTROLLED arm,
#         corrupting the AINOS3-86 0.00 % FP figure
#       - schema_audit.py:178 / audit_dead_columns.py:52  blanks are not
#         counted, so AINOS3-108's --require-no-new-silent gate passes anything
#
# Seeding every field with the sentinel and refusing to adopt it gives both.


def _stringify(value):
    """Convert a telemetry value to a CSV-friendly string.

    ⚠ This MUST stay identical to `csv_output_plugin._stringify`, because the
    blend decides what changed in the same string domain the CSV records and
    `deinterleave_csv.py` later reads back. Deliberately duplicated rather than
    imported: this module loads inside the OnAIR core package, before any
    plugin path is resolved. `test_stringify_matches_csv_output` pins the two
    together so the copy cannot drift.

    cFS char[] fields surface as Python `bytes` through ctypes; calling str()
    on them produces "b'X'" with the literal byte-string prefix. Decode
    instead, stripping nul-padding, so the CSV holds plain text.
    """
    if isinstance(value, bytes):
        return value.rstrip(b'\x00').decode('utf-8', errors='replace')
    return str(value)


class BlendEngine:
    """Per-buffer change detection over a fixed, positionally-indexed frame.

    Pure state machine: no I/O, no config, no clock. Flat lists rather than
    dicts because the frame layout is fixed at parse time and positional
    indexing keeps the per-frame cost a tight scan — this runs on every frame
    of a live pipeline.
    """

    def __init__(self, n_fields: int):
        # Reference copy of the last frame seen from each buffer, as strings.
        # None until that buffer has delivered its first frame.
        self._refs: list[list[str] | None] = [None, None]
        # The coherent output, seeded with the sentinel (see the note above).
        self._blend_s: list[str] = [SENTINEL] * n_fields
        # The same output as original objects, so `inline` mode can hand the
        # pipeline the value types it has always received instead of strings.
        # Seeded with a distinct [0] per field: _stringify([0]) == SENTINEL, so
        # the two representations agree even for a field nothing ever delivers.
        self._blend_o: list = [[0] for _ in range(n_fields)]
        self.n_fields = n_fields
        self.frames = 0
        self.updates = [0, 0]
        self.first_seen = [False, False]

    def feed(self, frame: list, buf: int):
        """Fold one frame from buffer `buf` into the blend.

        Returns `(blended_strings, blended_objects)` — live references to the
        engine's own lists, valid until the next `feed`. The caller must copy
        or consume them immediately; the CSV writer does the latter.
        """
        svals = [_stringify(v) for v in frame]
        ref = self._refs[buf]
        bs, bo = self._blend_s, self._blend_o

        if ref is None:
            # First frame from this buffer: nothing to diff against, so every
            # field it actually carries is adopted. This is what seeds the
            # output.
            for i, s in enumerate(svals):
                if s == SENTINEL:
                    continue
                bs[i] = s
                # Safe to hold the object: get_current_data REPLACES the list
                # element (`data[idx] = ...`), it never mutates in place.
                bo[i] = frame[i]
            self.first_seen[buf] = True
        else:
            n = 0
            for i, s in enumerate(svals):
                if s != ref[i]:
                    if s == SENTINEL:
                        continue
                    bs[i] = s
                    bo[i] = frame[i]
                    n += 1
            self.updates[buf] += n

        self._refs[buf] = svals
        self.frames += 1
        return bs, bo

    def stats(self) -> dict:
        return {
            "frames": self.frames,
            "updates_buf0": self.updates[0],
            "updates_buf1": self.updates[1],
            "first_seen": list(self.first_seen),
        }


class BlendedCsvWriter:
    """Writes the blended stream in `csv_output`'s exact format.

    ⚠ Same pruning, same quoting, same sidecar shape as
    `csv_output_plugin.Plugin` — deliberately mirrored rather than imported, for
    the same load-order reason as `_stringify`, and pinned by
    `test_writer_matches_csv_output_byte_for_byte`.

    One difference, and it is a deliberate improvement: the file handle is held
    open and flushed per row instead of reopened per row. Durability is
    identical (csv_output does not fsync either), and it removes ~2 syscalls per
    frame from the live path.
    """

    def __init__(self, headers, output_dir, filename_template, exclude_columns, meta_extras=None, flush_every=1):
        self.headers = list(headers)
        self.exclude_columns = set(exclude_columns or ())
        self.keep_indices = [i for i, h in enumerate(self.headers)
                             if h not in self.exclude_columns]
        self.filtered_headers = [self.headers[i] for i in self.keep_indices]
        self.output_dir = output_dir
        self.filename_template = filename_template
        self.meta_extras = dict(meta_extras or {})
        self.flush_every = max(1, int(flush_every))
        self.pid = os.getpid()
        self.file_name = None
        self.rows = 0
        self._fh = None
        self._w = None
        os.makedirs(self.output_dir, exist_ok=True)

    def _make_filename(self):
        ts = datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
        name = self.filename_template.format(timestamp=ts, pid=self.pid) + '.csv'
        return os.path.join(self.output_dir, name)

    def recorded_schema_sha256(self):
        """sha256 over the ordered column list actually written (post-prune).

        Newline-joined so a column rename cannot collide with a reordering.
        Matches csv_output's `_recorded_schema_sha256` so a blended file pins
        the same RECORDED schema its raw sibling does (AINOS3-124).
        """
        return hashlib.sha256('\n'.join(self.filtered_headers).encode('utf-8')).hexdigest()

    def _write_meta_sidecar(self):
        meta = {
            'csv_basename': os.path.basename(self.file_name),
            'pid': self.pid,
            'lines_per_file': 0,
            'excluded_columns': sorted(self.exclude_columns),
            'kept_columns_count': len(self.filtered_headers),
            'kept_columns': list(self.filtered_headers),
            'recorded_schema_sha256': self.recorded_schema_sha256(),
        }
        meta.update(self.meta_extras)
        path = self.file_name[:-4] + '.meta.json' if self.file_name.endswith('.csv') \
            else self.file_name + '.meta.json'
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)

    def write(self, svals):
        """Append one already-stringified frame."""
        if self._fh is None:
            self.file_name = self._make_filename()
            self._fh = open(self.file_name, 'a', newline='')
            self._w = csv.writer(self._fh)
            self._w.writerow(self.filtered_headers)
            self._fh.flush()
            self._write_meta_sidecar()
        self._w.writerow([svals[i] for i in self.keep_indices])
        self.rows += 1
        if self.rows % self.flush_every == 0:
            self._fh.flush()

    def close(self):
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            finally:
                self._fh = None
                self._w = None


class _Profiler:
    """Bounded per-frame cost sampler for the blend + write path.

    Exists because "the blend must not become a bottleneck" is a claim that
    needs evidence on the live pipeline, not just on a bench replay. Reports
    against the measured frame interval, so the number that matters — what
    fraction of a frame the transform consumes — is read directly.
    """

    def __init__(self, every: int, window: int = 2000):
        self.every = int(every)
        self.window = window
        self.blend_us = []
        self.write_us = []
        self.interval_ms = []
        self._last_t = None

    def sample(self, blend_us, write_us, now):
        if self._last_t is not None:
            self.interval_ms.append((now - self._last_t) * 1e3)
            if len(self.interval_ms) > self.window:
                del self.interval_ms[0]
        self._last_t = now
        self.blend_us.append(blend_us)
        self.write_us.append(write_us)
        if len(self.blend_us) > self.window:
            del self.blend_us[0]
            del self.write_us[0]

    @staticmethod
    def _pct(xs, p):
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]

    def report(self, frames):
        tot = [b + w for b, w in zip(self.blend_us, self.write_us)]
        iv = self._pct(self.interval_ms, 50)
        share = (self._pct(tot, 50) / 1e3 / iv * 100.0) if iv else 0.0
        return (
            f"[sbn_adapter_blended] frame {frames}: "
            f"blend p50 {self._pct(self.blend_us, 50):.0f}us "
            f"p95 {self._pct(self.blend_us, 95):.0f}us "
            f"max {max(self.blend_us or [0]):.0f}us | "
            f"write p50 {self._pct(self.write_us, 50):.0f}us "
            f"p95 {self._pct(self.write_us, 95):.0f}us | "
            f"frame interval p50 {iv:.0f}ms | "
            f"transform = {share:.2f}% of a frame"
        )


class ArrivalMeter:
    """How often each subscribed packet ACTUALLY arrives over SBN.

    Exists because the CSV cannot answer that question. A counter such as
    `CFE_ES.CommandCounter` is static in nominal flight, so its column changes
    zero times per second whether its packet lands at 0.25 Hz or at 5 Hz — yet
    that cadence is exactly what decides whether a fast attack (DE-0003.01's
    NOOP+RESET in 0.7 s) is visible at all. Arrivals are counted at the
    listener, before any buffering, blending, or frame-rate cap.

    ⚠ Log-only by design. Nothing here enters the frame, so the recorded
    schema (and `recorded_schema_sha256`, the AINOS3-124 freeze) is untouched.

    `record()` runs on the SBN listener thread and `report()` on the OnAIR
    main thread, hence the lock. Each report covers the window since the
    previous one and appends one JSON line to `arrivals_<ts>_pid<N>.jsonl`.
    """

    def __init__(self, every_s, output_dir, clock=time.monotonic):
        self.every_s = float(every_s)
        self.output_dir = output_dir
        self.clock = clock
        self.pid = os.getpid()
        self.file_name = None
        self._lock = threading.Lock()
        self._total = {}
        self._n = {}
        self._gaps = {}
        self._last = {}
        self._win_start = clock()

    def register(self, packets):
        """Pre-seed packets so one that NEVER arrives still reports n = 0."""
        with self._lock:
            for p in packets:
                self._total.setdefault(p, 0)
                self._n.setdefault(p, 0)
                self._gaps.setdefault(p, [])

    def record(self, packet, now=None):
        now = self.clock() if now is None else now
        with self._lock:
            self._total[packet] = self._total.get(packet, 0) + 1
            self._n[packet] = self._n.get(packet, 0) + 1
            last = self._last.get(packet)
            if last is not None:
                self._gaps.setdefault(packet, []).append(now - last)
            else:
                self._gaps.setdefault(packet, [])
            self._last[packet] = now

    def due(self, now=None):
        now = self.clock() if now is None else now
        return self.every_s > 0 and (now - self._win_start) >= self.every_s

    def snapshot(self, now=None):
        """Stats for the window since the last snapshot; starts a new window."""
        now = self.clock() if now is None else now
        with self._lock:
            window = now - self._win_start
            packets = {}
            for p in sorted(self._total):
                gaps = sorted(self._gaps.get(p, []))
                last = self._last.get(p)
                packets[p] = {
                    'n': self._n.get(p, 0),
                    'hz': round(self._n.get(p, 0) / window, 3) if window > 0 else None,
                    'gap_p50_ms': round(gaps[len(gaps) // 2] * 1e3, 1) if gaps else None,
                    'gap_max_ms': round(gaps[-1] * 1e3, 1) if gaps else None,
                    # A packet that has STOPPED shows here, not in gap_max_ms,
                    # which only sees gaps that have already closed.
                    'since_last_ms': round((now - last) * 1e3, 1) if last is not None else None,
                    'total': self._total[p],
                }
                self._n[p] = 0
                self._gaps[p] = []
            self._win_start = now
        return {'utc': datetime.now(timezone.utc).isoformat(),
                'window_s': round(window, 3), 'packets': packets}

    def write(self, snap):
        if self.file_name is None:
            os.makedirs(self.output_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
            self.file_name = os.path.join(self.output_dir, f"arrivals_{ts}_pid{self.pid}.jsonl")
        with open(self.file_name, 'a') as f:
            f.write(json.dumps(snap) + '\n')

    @staticmethod
    def summary(snap):
        by_hz = sorted(snap['packets'].items(), key=lambda kv: (kv[1]['hz'] or 0.0))
        body = ", ".join(f"{p} {s['hz']:.2f}" for p, s in by_hz if s['hz'] is not None)
        return f"[sbn_adapter_blended] arrivals Hz over {snap['window_s']:.1f}s: {body}"

    def report(self, now=None):
        snap = self.snapshot(now)
        self.write(snap)
        print(self.summary(snap), flush=True)
        return snap


def resolve_arrival_dir(configured, csv_output_dir):
    """Blank -> a SIBLING `arrivals/` of csv_output's dir, for the same reason
    `resolve_blended_dir` keeps blended output out of `csv/`."""
    if configured and str(configured).strip():
        return str(configured).strip()
    parent = os.path.dirname(csv_output_dir.rstrip('/')) or '.'
    return os.path.normpath(os.path.join(parent, 'arrivals'))


def _ini_path():
    return os.environ.get('ONAIR_INI_FILE') or 'cf/onair/nos3_security.ini'


def resolve_blended_dir(configured, csv_output_dir):
    """Where the blended stream lands. Blank -> a SIBLING `csv_blended/`.

    ⚠ Not `csv_output`'s own directory, and the reason is not tidiness:

      * `AINOS3-125 AC4` already put the offline-blended corpus in
        `data/csv_blended/`, keeping each file's `csv_out_*` basename.
        The DIRECTORY carries the meaning, so native output keeps the same
        basename and every tool globbing `csv_out_*.csv` — `loader.load`,
        `list_clean_csvs`, `build_corpus_manifest` — works on either
        directory with no change and no filename-prefix special case.
      * `data/csv` holds 11 recording generations AND every plugin
        side-file (`attack_class_`, `iforest_out_`, `rule_gate_out_`, ...).
        Adding a fourth filename family to the pile `AINOS3-97` exists to
        quarantine makes that job harder, and any tool that globs `*.csv`
        rather than `csv_out_*.csv` would silently pick blended frames up.
      * The blended stream roughly doubles recorded volume. Its own directory
        is independently prunable and independently quarantinable.
    """
    if configured and str(configured).strip():
        return str(configured).strip()
    parent = os.path.dirname(csv_output_dir.rstrip('/')) or '.'
    return os.path.normpath(os.path.join(parent, 'csv_blended'))


def load_adapter_config(ini_path=None):
    """Read `[SBN_ADAPTER]`, falling back to `[CSV_OUTPUT]` for the shared keys.

    The exclusion list is SHARED with csv_output on purpose: the blended file
    must be pruned exactly like its raw sibling, or `deinterleave_csv.py(raw)`
    and the blended file would not have the same columns to compare.

    ⚠ The output DIRECTORY is deliberately NOT shared. Blank `BlendedOutputDir`
    resolves to a sibling `csv_blended/` of csv_output's dir — see
    `resolve_blended_dir`.
    """
    path = ini_path or _ini_path()
    cfg = {
        'blendmode': 'tap',
        'blendedfilenametemplate': 'csv_out_{timestamp}_pid{pid}',
        'blendedoutputdir': None,
        'profileevery': '0',
        'flushevery': '1',
        'frametimestamp': 'true',
        'simtime': 'true',
        'arrivalreportevery': '0',
        'arrivaloutputdir': None,
        'outputdir': 'data/csv',
        'excludecolumns': '',
        'csvfilenametemplate': 'csv_out_{timestamp}_pid{pid}',
        'schema_path': None,
        'schema_sha256': None,
    }
    if not os.path.exists(path):
        return cfg
    parser = configparser.ConfigParser()
    parser.read(path)
    if parser.has_section('CSV_OUTPUT'):
        co = dict(parser.items('CSV_OUTPUT'))
        cfg['outputdir'] = co.get('outputdir', cfg['outputdir'])
        cfg['excludecolumns'] = co.get('excludecolumns', '')
        cfg['csvfilenametemplate'] = co.get('filenametemplate', cfg['csvfilenametemplate'])
    if parser.has_section('SBN_ADAPTER'):
        sa = dict(parser.items('SBN_ADAPTER'))
        for k in ('blendmode', 'blendedfilenametemplate', 'blendedoutputdir', 'profileevery', 'flushevery', 'frametimestamp', 'frameintervalms',
                  'arrivalreportevery', 'arrivaloutputdir'):
            if k in sa:
                cfg[k] = sa[k]
    if parser.has_section('FILES'):
        mp = parser.get('FILES', 'MetaFilePath', fallback='').strip()
        mf = parser.get('FILES', 'MetaFile', fallback='').strip()
        if mf:
            rel = os.path.join(mp, mf) if mp else mf
            cfg['schema_path'] = rel
            if os.path.exists(rel):
                with open(rel, 'rb') as f:
                    cfg['schema_sha256'] = hashlib.sha256(f.read()).hexdigest()
    return cfg


class DataSource(sbn_adapter.DataSource):
    """SBN adapter that reconstructs one coherent frame from the double buffer.

    Subclasses rather than replaces `sbn_adapter.DataSource`, and does not
    override `get_next()`'s buffer logic — it calls it. That is AINOS3-126 AC2b
    by construction: in `tap` mode the raw stream is literally the parent's
    code path, so old and new `csv/` data are the same representation and pool
    into one corpus.
    """

    VALID_MODES = ('tap', 'inline', 'off')

    def __init__(self, data_file, meta_file, ss_breakdown=False):
        # Read before super().__init__ — it calls connect(), which starts the
        # listener thread, and nothing below should race that.
        self._cfg = load_adapter_config()
        self._stamp_frames = str(
            self._cfg.get('frametimestamp', 'true')).strip().lower() == 'true'
        self._ts_idx = None
        self._stamp_simtime = str(
            self._cfg.get('simtime', 'true')).strip().lower() == 'true'
        self._sim_idx = None
        self._sim_src = None        # (weeks_idx, sow_idx, leap_idx)
        self._sim_clock = None
        mode = str(self._cfg.get('blendmode', 'tap')).strip().lower()
        if mode not in self.VALID_MODES:
            print(f"[sbn_adapter_blended] WARNING: unknown BlendMode '{mode}'; falling back to 'tap'")
            mode = 'tap'
        self.blend_mode = mode
        self._blend = None
        self._writer = None
        self._frames = 0
        try:
            prof_every = int(self._cfg.get('profileevery', '0'))
        except ValueError:
            prof_every = 0
        self._prof = _Profiler(prof_every) if prof_every > 0 else None
        self._warned_len = False

        # Frame-rate cap. The recorder writes one row per get_next(), and
        # get_next() returns as soon as ANY subscribed message has arrived, so
        # the row rate rides the aggregate SBN arrival rate (measured ~6 Hz at
        # 1 Hz sensors, ~16 Hz at 10 Hz sensors). That makes columns whose MID
        # publishes slower than the row rate repeat for several rows. Capping
        # the frame rate to a fixed cadence and publishing every dynamic MID
        # ABOVE that cadence (oversample-then-decimate) makes each emitted row
        # land on fresh data. FrameIntervalMs = 0 disables the cap (stock
        # arrival-driven behaviour); 200 -> a steady 5 Hz.
        try:
            self._frame_interval_s = max(
                0.0, float(self._cfg.get('frameintervalms', '0')) / 1000.0)
        except ValueError:
            self._frame_interval_s = 0.0
        self._last_emit = None

        # Built BEFORE super().__init__: connect() starts the listener thread,
        # which calls get_current_data() -> self._arrivals.record().
        try:
            arrival_every = float(self._cfg.get('arrivalreportevery', '0'))
        except ValueError:
            arrival_every = 0.0
        self._arrivals = ArrivalMeter(
            arrival_every,
            resolve_arrival_dir(self._cfg.get('arrivaloutputdir'),
                                self._cfg.get('outputdir'))) if arrival_every > 0 else None

        super().__init__(data_file, meta_file, ss_breakdown)

        if self._arrivals is not None:
            self._arrivals.register(v[0] for v in self.msgID_lookup_table.values())
            print(f"[sbn_adapter_blended] arrival meter every {arrival_every:g}s -> "
                  f"{self._arrivals.output_dir}")

        print(f"[sbn_adapter_blended] {ADAPTER_VERSION} BlendMode={self.blend_mode}")
        if self.blend_mode == 'inline':
            # In inline mode csv_output records the BLENDED stream. The
            # DIRECTORY is what marks data as blended, so if OutputDir still
            # points at the raw csv/ dir the corpus would carry blended frames
            # in the interleaved corpus with nothing to distinguish them.
            # Loud, because it is silent otherwise.
            out = str(self._cfg.get('outputdir', ''))
            if 'csv_blended' not in out:
                print("[sbn_adapter_blended] ⚠ WARNING: BlendMode=inline but "
                      f"[CSV_OUTPUT] OutputDir is '{out}' — the frames it "
                      "writes are BLENDED and would land in the interleaved "
                      "corpus. Repoint OutputDir (and every plugin's "
                      "SideFileOutputDir) at csv_blended/.")
            print("[sbn_adapter_blended] ⚠ inline mode changes what the "
                  "DETECTORS read. The IF and classifier must be refitted on "
                  "blended data (AINOS3-126 AC4) or they are scoring a "
                  "distribution they were never fitted to.")

    def parse_meta_data_file(self, meta_data_file, ss_breakdown):
        """Append the timestamp column to the frame layout and the binning.

        ⚠ Done HERE, not in `nos3_security_tlm.json`, and the distinction is
        load-bearing: putting it in the tlm file would make the recorded layout
        depend on a column the STOCK `sbn_adapter` does not produce, so running
        the stock adapter against that schema would hand `csv_output` a row one
        shorter than its headers and silently mis-label every column. Keeping it
        in this subclass means the tlm file — and therefore `schema_sha256` —
        is untouched, and only `recorded_schema_sha256` moves, which is correct:
        the RECORDED schema really does gain a column.

        ⚠ `vehicle_rep.py:24` asserts `len(headers) == len(tests)`, so every
        parallel binning list grows too, not just the labels.
        """
        configs = super().parse_meta_data_file(meta_data_file, ss_breakdown)

        def _add(col, desc):
            for buf in self.currentData:
                buf['headers'].append(col)
                buf['data'].append("")
            configs['data_labels'].append(col)
            configs['subsystem_assignments'].append(['NONE'])
            configs['test_assignments'].append([['NOOP']])
            configs['description_assignments'].append(desc)
            return len(self.currentData[0]['headers']) - 1

        if self._stamp_frames:
            self._ts_idx = _add(
                TIMESTAMP_COLUMN,
                "OnAIR frame arrival time (ISO-8601 UTC), stamped when "
                "get_next() returns the frame. Same form as an attack "
                "manifest's start_utc.")

        if self._stamp_simtime:
            hdrs = self.currentData[0]['headers']
            try:
                self._sim_src = (hdrs.index(_GPS_WEEKS), hdrs.index(_GPS_SOW),
                                 hdrs.index(_LEAP_SECONDS))
            except ValueError:
                # No GPS in this schema — do not emit a column we cannot fill.
                print(f"[sbn_adapter_blended] ⚠ {SIM_TIME_COLUMN} SKIPPED: "
                      f"{_GPS_WEEKS}/{_GPS_SOW} not in the subscribed schema")
                self._stamp_simtime = False
            else:
                epoch = read_sim_epoch()
                if epoch is None:
                    print(f"[sbn_adapter_blended] ⚠ could not read the sim epoch "
                          f"from cfg/InOut/Inp_Sim.txt; {SIM_TIME_COLUMN} will "
                          f"stay blank rather than emit a date off by a "
                          f"multiple of ~19.6 years (GPS week rollover)")
                else:
                    print(f"[sbn_adapter_blended] sim epoch {epoch.isoformat()} "
                          f"(cfg/InOut/Inp_Sim.txt)")
                self._sim_clock = SimClock(epoch)
                self._sim_idx = _add(
                    SIM_TIME_COLUMN,
                    "Simulation time (ISO-8601 UTC) decoded from the NOVATEL "
                    "GPS week/second pair, with the 10-bit week rollover "
                    "resolved against cfg/InOut/Inp_Sim.txt. Monotonic: torn "
                    "reads are rejected and the last good value carried.")
        return configs

    def _init_blend(self, n_fields):
        self._blend = BlendEngine(n_fields)

        headers = list(self.all_headers)
        if len(headers) != n_fields:
            # Positional alignment between all_headers and the frame is assumed
            # by csv_output too; if it is broken, the blended file's columns
            # would be mislabelled. Do not write a file we cannot trust.
            print(f"[sbn_adapter_blended] ⚠ ERROR: all_headers has "
                  f"{len(headers)} entries but the frame has {n_fields}; "
                  f"blended side-file DISABLED (blending still active).")
            return

        if self.blend_mode != 'tap':
            # inline: csv_output writes the blended stream, so no side file.
            return

        excl = {c.strip() for c in str(self._cfg.get('excludecolumns', '')).split(',') if c.strip()}
        out_dir = resolve_blended_dir(self._cfg.get('blendedoutputdir'), self._cfg.get('outputdir'))
        self._writer = BlendedCsvWriter(
            headers=headers,
            output_dir=out_dir,
            filename_template=self._cfg.get('blendedfilenametemplate'),
            exclude_columns=excl,
            flush_every=self._cfg.get('flushevery', 1),
            meta_extras={
                'schema_path': self._cfg.get('schema_path'),
                'schema_sha256': self._cfg.get('schema_sha256'),
                'container': socket.gethostname(),
                'plugin_version': ADAPTER_VERSION,
                # Provenance, per AINOS3-125 AC4. `native` distinguishes a
                # file the adapter blended live from one deinterleave_csv.py
                # produced offline — they must be identical, and the whole
                # point of AC2 is proving it, so the two must be tellable
                # apart in the corpus.
                'transform': 'deinterleave/per-buffer-change-detection',
                'transform_source': 'native (sbn_adapter_blended)',
                'buffer_assignment': 'authoritative (double_buffer_read_index)',
            },
        )
        print(f"[sbn_adapter_blended] blended stream -> {self._writer._make_filename()} (template)")

    def get_current_data(self, recv_msg, data_struct, app_name):
        """Count the arrival, then hand the packet to the parent unchanged."""
        if self._arrivals is not None:
            self._arrivals.record(app_name)
        return super().get_current_data(recv_msg, data_struct, app_name)

    def get_next(self):
        """One coherent frame, or the raw frame with the blend side-written.

        ⚠ The parent's `get_next()` is CALLED, not reimplemented. The raw read
        path — the buffer flip, the lock, the wait — is untouched.
        """
        # Pace BEFORE reading so the frame we return is the freshest available
        # at the cap instant, not one held from the top of the interval. The
        # cap is a ceiling only: super().get_next() still blocks for data, so a
        # slow stream simply emits slower than the cap.
        if self._frame_interval_s > 0.0 and self._last_emit is not None:
            wait = self._frame_interval_s - (time.monotonic() - self._last_emit)
            if wait > 0.0:
                time.sleep(wait)

        frame = super().get_next()
        self._last_emit = time.monotonic()

        if self._arrivals is not None and self._arrivals.due():
            self._arrivals.report()

        # ⚠ Stamped on the frame that is being RETURNED, so the raw file
        # (written by csv_output) and the blended file (written here) carry the
        # SAME value for the same row — one clock reading, two records of it.
        # It also survives the blend untouched: the value differs from this
        # buffer's previous frame every time, so the change-detection rule
        # adopts it on every row, which is exactly right.
        if self._ts_idx is not None and self._ts_idx < len(frame):
            frame[self._ts_idx] = datetime.now(timezone.utc).isoformat()

        # ⚠ Derived from the RAW frame but made monotonic by SimClock's own
        # state, so the value written here is already clean. That is what keeps
        # the raw and blended files identical in this column.
        if self._sim_idx is not None and self._sim_idx < len(frame):
            w, sw, lp = self._sim_src
            frame[self._sim_idx] = self._sim_clock.update(
                frame[w], frame[sw], frame[lp])

        if self.blend_mode == 'off':
            return frame

        if self._blend is None:
            self._init_blend(len(frame))

        t0 = time.perf_counter()
        # Authoritative buffer identity: we KNOW which buffer this frame came
        # from, where the offline script has to infer it from strict
        # alternation and check the inference. No parity-disagreement class of
        # error exists here.
        bs, bo = self._blend.feed(frame, self.double_buffer_read_index)
        t1 = time.perf_counter()

        if self._writer is not None:
            self._writer.write(bs)
        t2 = time.perf_counter()

        self._frames += 1
        if self._prof is not None:
            self._prof.sample((t1 - t0) * 1e6, (t2 - t1) * 1e6, t2)
            if self._frames % self._prof.every == 0:
                print(self._prof.report(self._frames), flush=True)

        return bo if self.blend_mode == 'inline' else frame
