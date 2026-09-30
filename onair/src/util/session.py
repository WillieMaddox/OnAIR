"""One directory per OnAIR session — the OnAIR half of the contract.

The other half is `ainos3/training/scenarios/sessions.py`; the two share the file layout and the
session.json format and must stay in step.

At startup the data source (sbn_adapter_blended) calls `claim()`. If the ini sets
`[SESSION] SessionsDir`, OnAIR takes `SessionsDir/.next_session.json` — written by the harness
before `make launch-quiet` — by renaming it into a new `SessionsDir/<session_id>/session.json`. A
launch with no (or a stale) claim file gets `<YYYY-MM-DDTHH-MM-SSZ>_adhoc`. Every plugin then asks
`path(<key>)` for its output file instead of choosing a directory and a timestamp itself, so one
session writes one directory with fixed file names.

With `SessionsDir` blank or absent the module is inert (`current()` is None) and every writer keeps
its legacy `<dir>/<name>_<ts>_pid<N>.csv` behaviour: rolling back is one ini line.

session.json has two writers (this process and the harness on the host), so writes are a locked
read-modify-write (`flock` on `.session.lock`) and an atomic replace.
"""
from __future__ import annotations

import configparser
import datetime as dt
import fcntl
import json
import os
import socket

CLAIM_NAME = ".next_session.json"
CURRENT_NAME = ".current_session"
LOCK_NAME = ".session.lock"
SESSION_JSON = "session.json"
# A claim older than this was left by a harness that died before launching. Longer than the
# harness allows `make launch-quiet` (600 s), so a slow launch still claims its session.
CLAIM_MAX_AGE_S = 900

FILES = {
    "raw": "raw.csv",
    "raw_meta": "raw.meta.json",
    "blended": "blended.csv",
    "blended_meta": "blended.meta.json",
    "arrivals": "arrivals.jsonl",
}
DETECTORS = ("iforest", "attack_class", "incident", "rule_gate", "rule_gate_incident",
             "consistency", "consistency_incident", "staleness", "staleness_incident")
# Detector outputs that are not CSV.
DETECTOR_EXT = {"iforest_golden": "npz"}

_current = None          # the claimed Session for this process, once claimed
_resolved = False        # claim() has run (successfully or inert)


def _utc_now():
    return dt.datetime.now(dt.timezone.utc)


def _iso(t):
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ini_path():
    return os.environ.get("ONAIR_INI_FILE") or "cf/onair/nos3_security.ini"


def sessions_dir_from_ini(ini_path=None):
    """[SESSION] SessionsDir, or None when unset (legacy per-plugin output)."""
    path = ini_path or _ini_path()
    if not os.path.exists(path):
        return None
    parser = configparser.ConfigParser()
    parser.read(path)
    val = parser.get("SESSION", "SessionsDir", fallback="").strip()
    return val or None


def _write_json_atomic(path, obj):
    tmp = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _deep_merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


class Session:
    def __init__(self, directory):
        self.dir = directory
        self.id = os.path.basename(directory.rstrip("/"))

    def __repr__(self):
        return f"Session({self.id})"

    @property
    def json_path(self):
        return os.path.join(self.dir, SESSION_JSON)

    def read(self):
        try:
            with open(self.json_path) as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {}

    def update(self, fields):
        fd = os.open(os.path.join(self.dir, LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            rec = self.read()
            _deep_merge(rec, fields)
            _write_json_atomic(self.json_path, rec)
            return rec
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def path(self, key):
        """Fixed output path for `key` (a FILES key or a detector name); creates parents."""
        if key in FILES:
            p = os.path.join(self.dir, FILES[key])
        elif key in DETECTORS or key in DETECTOR_EXT:
            p = os.path.join(self.dir, "detectors", f"{key}.{DETECTOR_EXT.get(key, 'csv')}")
        else:
            raise KeyError(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def register(self, key, path, **fields):
        """Record a file this process writes, plus any extra top-level fields."""
        rel = os.path.relpath(path, self.dir)
        upd = {"files": {key: rel}}
        upd.update(fields)
        self.update(upd)


def _unique_dir(root, sid):
    cand, n = sid, 1
    while os.path.exists(os.path.join(root, cand)):
        n += 1
        cand = f"{sid}-{n}"
    return cand


def _read_claim(claim_path, now):
    """The claim record if it is usable; (None, reason) otherwise."""
    try:
        with open(claim_path) as fh:
            rec = json.load(fh)
    except FileNotFoundError:
        return None, "no claim file"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable claim file ({exc})"
    sid = rec.get("session_id")
    if not sid or "/" in sid or sid.startswith("."):
        return None, f"claim has a bad session_id {sid!r}"
    try:
        minted = dt.datetime.strptime(rec["minted_utc"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=dt.timezone.utc)
    except (KeyError, ValueError):
        return None, "claim has no minted_utc"
    age = (now - minted).total_seconds()
    if age > CLAIM_MAX_AGE_S:
        return None, f"claim {sid} is stale ({age:.0f}s old > {CLAIM_MAX_AGE_S}s)"
    return rec, None


def claim(sessions_dir=None, now=None):
    """Claim this process's session. Idempotent; returns the Session, or None when inert."""
    global _current, _resolved
    if _resolved:
        return _current
    root = sessions_dir if sessions_dir is not None else sessions_dir_from_ini()
    _resolved = True
    if not root:
        return None
    now = now or _utc_now()
    os.makedirs(root, exist_ok=True)
    claim_path = os.path.join(root, CLAIM_NAME)
    rec, why = _read_claim(claim_path, now)
    how = "harness"
    if rec is not None:
        sid = rec["session_id"]
        sdir = os.path.join(root, sid)
        try:
            os.makedirs(sdir)                                    # fails if it already exists
            os.rename(claim_path, os.path.join(sdir, SESSION_JSON))   # the atomic claim
        except FileExistsError:
            rec, why = None, f"session directory {sid} already exists"
        except FileNotFoundError:
            # Another process renamed it first. Leave the empty dir for it; take adhoc.
            rec, why = None, f"claim {sid} taken by another process"
    if rec is None:
        how = "adhoc"
        sid = _unique_dir(root, now.strftime("%Y-%m-%dT%H-%M-%SZ") + "_adhoc")
        sdir = os.path.join(root, sid)
        os.makedirs(sdir)
        _write_json_atomic(os.path.join(sdir, SESSION_JSON), {
            "session_id": sid, "purpose": "adhoc", "technique": None, "mode": None,
            "status": "new", "status_reason": None,
            "referenced_by": {"corpora": [], "anomalies": []},
        })
        print(f"[session] no usable claim ({why}) -> {sid}")
    else:
        print(f"[session] claimed {sid}")
    s = Session(sdir)
    s.update({
        "started_utc": _iso(now),
        "onair": {"claim": how, "claimed_utc": _iso(now), "pid": os.getpid(),
                  "container": socket.gethostname(), "ini": os.path.abspath(_ini_path()),
                  **({"claim_skipped": why} if how == "adhoc" else {})},
    })
    _write_json_atomic_text(os.path.join(root, CURRENT_NAME), sid + "\n")
    _current = s
    return s


def _write_json_atomic_text(path, text):
    tmp = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def current():
    """This process's session, claiming it on first use (a plugin may run before the adapter in
    tests or with the stock adapter). None when [SESSION] SessionsDir is unset."""
    return _current if _resolved else claim()


def path(key):
    """Output path for `key` in this process's session, or None when sessions are off."""
    s = current()
    return s.path(key) if s is not None else None


def side_file(key, legacy_dir, legacy_stem, ext="csv"):
    """Where a plugin writes its side-file `key`.

    In a session: <session>/detectors/<key>.<ext>. Otherwise the legacy flat name
    `<legacy_dir>/<legacy_stem>_<ts>_pid<N>.<ext>` (creating legacy_dir). The file is not
    registered here — the harness records the files that actually exist when it finalizes.
    """
    s = current()
    if s is not None:
        return s.path(key)
    os.makedirs(legacy_dir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    return os.path.join(legacy_dir, f"{legacy_stem}_{ts}_pid{os.getpid()}.{ext}")


def register(key, file_path, **fields):
    s = current()
    if s is not None:
        s.register(key, file_path, **fields)


def _reset_for_tests():
    global _current, _resolved
    _current, _resolved = None, False
