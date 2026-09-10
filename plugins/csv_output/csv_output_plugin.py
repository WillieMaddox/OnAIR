# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import configparser
import csv
import hashlib
import json
import os
import socket
from datetime import datetime
from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin


PLUGIN_VERSION = "csv_output@1.2"


def _stringify(value):
    """Convert a telemetry value to a CSV-friendly string.

    cFS char[] fields surface as Python `bytes` through ctypes; calling str()
    on them produces "b'X'" with the literal byte-string prefix. Decode
    instead, stripping nul-padding, so the CSV holds plain text.
    """
    if isinstance(value, bytes):
        return value.rstrip(b'\x00').decode('utf-8', errors='replace')
    return str(value)


class Plugin(AIPlugin):
    """
    CSV writer for OnAIR telemetry frames.

    Configurable via the [CSV_OUTPUT] section of the OnAIR ini file. Path
    discovery: ONAIR_INI_FILE env var, then 'cf/onair/nos3_security.ini'
    relative to cwd. If neither is found, defaults below are used.

    Filename template tokens (Python str.format):
        {timestamp} - ISO 8601 local time, : -> -    e.g. 2026-04-30T12-08-23
        {pid}       - Python process id (int)

    LinesPerFile = 0 disables rotation entirely — one CSV per OnAIR plugin
    lifetime (recommended). Set a positive integer only for tests or short
    capture windows; record-count rotation produces unmanageable file
    counts and boundaries that don't align with operational events.

    ExcludeColumns is a comma-separated list of header names to omit from
    the CSV output. Lets the operator prune fields the IF can't use (boot-
    time constants, filename strings) without changing the upstream ctypes
    struct definitions.

    A sidecar `<filename>.meta.json` is written at file creation recording
    session ID, two schema fingerprints, plugin version, container hostname,
    pid, and the exclusion list. Loaders can use this to verify schema
    consistency across files.

    The two fingerprints are NOT interchangeable (AINOS3-124):
      * `schema_sha256`          - sha256 of the active tlm metadata FILE.
        Changes on any edit to that file, including comment/ordering churn
        that does not move a column, and does NOT change when the ini's
        ExcludeColumns changes. It fingerprints the SUBSCRIBED schema.
      * `recorded_schema_sha256` - sha256 over the ordered list of columns
        actually written to this CSV (post-prune). This is the RECORDED
        schema: it is what a corpus is trained against, and it is the value
        a corpus manifest should pin.
    """

    DEFAULTS = {
        'OutputDir': 'data/onair/csv',
        'FilenameTemplate': 'csv_out_{timestamp}_pid{pid}',
        'LinesPerFile': '0',
        'WriteHeaderOnRotation': 'true',
        'ExcludeColumns': '',
    }

    def __init__(self, name, headers):
        super().__init__(name, headers)

        cfg = self._load_config()
        # ConfigParser lowercases option keys; honor that.
        self.output_dir = cfg.get('outputdir', self.DEFAULTS['OutputDir'])
        self.filename_template = cfg.get('filenametemplate', self.DEFAULTS['FilenameTemplate'])
        self.lines_per_file = int(cfg.get('linesperfile', self.DEFAULTS['LinesPerFile']))
        self.write_header_on_rotation = (
            cfg.get('writeheaderonrotation', self.DEFAULTS['WriteHeaderOnRotation']).strip().lower() == 'true'
        )
        excl_raw = cfg.get('excludecolumns', self.DEFAULTS['ExcludeColumns'])
        self.exclude_columns = {c.strip() for c in excl_raw.split(',') if c.strip()}

        self.pid = os.getpid()
        self.headers_built = False
        self.headers_written_to_current_file = False
        self.lines_current = 0
        self.current_buffer = []
        self.file_name = None  # set lazily on first render_reasoning()
        self.keep_indices = None  # computed lazily once headers are stable
        self.filtered_headers = None

        # Schema fingerprint + container info for the metadata sidecar.
        self._meta_extras = self._build_meta_extras()

        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _load_config():
        """Look up the [CSV_OUTPUT] section in the active OnAIR ini, if any."""
        ini_path = os.environ.get('ONAIR_INI_FILE') or 'cf/onair/nos3_security.ini'
        if not os.path.exists(ini_path):
            return {}
        parser = configparser.ConfigParser()
        parser.read(ini_path)
        if parser.has_section('CSV_OUTPUT'):
            return dict(parser.items('CSV_OUTPUT'))
        return {}

    @staticmethod
    def _build_meta_extras():
        """Read [FILES] to find the tlm metadata file, hash it, capture host info.

        The schema_sha256 lets a downstream loader detect that two CSVs were
        written under different schemas (e.g. nos3_security_tlm.json edits or
        a column-exclusion change) without having to diff every header.
        """
        ini_path = os.environ.get('ONAIR_INI_FILE') or 'cf/onair/nos3_security.ini'
        schema_sha = None
        meta_rel = None
        if os.path.exists(ini_path):
            parser = configparser.ConfigParser()
            parser.read(ini_path)
            if parser.has_section('FILES'):
                meta_path = parser.get('FILES', 'MetaFilePath', fallback='').strip()
                meta_file = parser.get('FILES', 'MetaFile', fallback='').strip()
                if meta_file:
                    meta_rel = os.path.join(meta_path, meta_file) if meta_path else meta_file
                    if os.path.exists(meta_rel):
                        with open(meta_rel, 'rb') as f:
                            schema_sha = hashlib.sha256(f.read()).hexdigest()
        return {
            'schema_path': meta_rel,
            'schema_sha256': schema_sha,
            'container': socket.gethostname(),
            'plugin_version': PLUGIN_VERSION,
        }

    def _recorded_schema_sha256(self):
        """sha256 over the ordered column list actually written to the CSV.

        Why this exists alongside `schema_sha256` (AINOS3-124): that one hashes
        the tlm metadata FILE, so it moves on comment churn and — the defect this
        fixes — does NOT move when ExcludeColumns changes. A prune is a real
        schema change to every downstream consumer, so the freeze and the corpus
        manifest need a fingerprint over the columns as recorded.

        Newline-joined so a column rename cannot collide with a reordering.
        """
        if self.filtered_headers is None:
            return None
        payload = '\n'.join(self.filtered_headers).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def _make_filename(self):
        # Microsecond precision so a rapid rotation cycle (e.g. small
        # LinesPerFile in tests, or a high-throughput scenario) cannot
        # produce two files with the same name within the same process.
        ts = datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
        name = self.filename_template.format(timestamp=ts, pid=self.pid) + '.csv'
        return os.path.join(self.output_dir, name)

    def _write_meta_sidecar(self):
        """Drop a <filename>.meta.json sidecar at file creation."""
        if self.file_name is None or self.filtered_headers is None:
            return
        meta = {
            'csv_basename': os.path.basename(self.file_name),
            'pid': self.pid,
            'lines_per_file': self.lines_per_file,
            'excluded_columns': sorted(self.exclude_columns),
            'kept_columns_count': len(self.filtered_headers),
            'kept_columns': list(self.filtered_headers),
            'recorded_schema_sha256': self._recorded_schema_sha256(),
        }
        meta.update(self._meta_extras)
        meta_path = self.file_name[:-4] + '.meta.json' if self.file_name.endswith('.csv') \
            else self.file_name + '.meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def update(self, low_level_data=[], high_level_data={}):
        """Stage one telemetry frame; high-level plugin names are appended to headers once.

        Note: csv_output runs as a knowledge_rep plugin, so vehicle_rep calls
        construct.update(frame) with a single arg — high_level_data is always
        the empty default dict here. Capturing learner outputs into the CSV
        therefore requires either moving csv_output to the complex-reasoning
        tier or adding a separate side-file writer; see the IF plugin notes.
        """
        if not self.headers_built:
            for layer in high_level_data.keys():
                for plugin in high_level_data[layer]:
                    self.headers.append(str(plugin))
            self.headers_built = True

        self.current_buffer = []
        for telem_point in low_level_data:
            self.current_buffer.append(_stringify(telem_point))
        for layer in high_level_data.keys():
            for plugin in high_level_data[layer]:
                plugin_output = high_level_data[layer].get(plugin, [])
                for telem_point in (plugin_output or []):
                    self.current_buffer.append(_stringify(telem_point))

    def render_reasoning(self):
        """Flush the staged frame to disk; rotate to a new file when LinesPerFile is hit.

        Uses csv.writer (QUOTE_MINIMAL) so values containing commas, quotes,
        or newlines are properly escaped. EVS event messages routinely have
        embedded commas, e.g. "Events squelched, AppName = LC".
        """
        if self.file_name is None:
            self.file_name = self._make_filename()

        # Lazy-compute the exclusion mask once the header list is stable
        # (i.e. after the first update()). When ExcludeColumns is empty
        # this reduces to identity — keep_indices == range(len(headers)).
        if self.keep_indices is None:
            self.keep_indices = [i for i, h in enumerate(self.headers)
                                 if h not in self.exclude_columns]
            self.filtered_headers = [self.headers[i] for i in self.keep_indices]

        if not self.headers_written_to_current_file:
            with open(self.file_name, 'a', newline='') as f:
                csv.writer(f).writerow(self.filtered_headers)
            self.headers_written_to_current_file = True
            # First write into this file — also drop the metadata sidecar.
            self._write_meta_sidecar()

        with open(self.file_name, 'a', newline='') as f:
            filtered_row = [self.current_buffer[i] for i in self.keep_indices]
            csv.writer(f).writerow(filtered_row)
        self.current_buffer = []
        self.lines_current += 1

        if self.lines_per_file != 0 and self.lines_current >= self.lines_per_file:
            self.file_name = self._make_filename()
            self.lines_current = 0
            # If WriteHeaderOnRotation is true, leave headers_written False so
            # the next render_reasoning() emits a header (and meta sidecar).
            # Otherwise, suppress both.
            self.headers_written_to_current_file = not self.write_header_on_rotation
