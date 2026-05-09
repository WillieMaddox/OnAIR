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
import os
from datetime import datetime
from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin


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

    Defaults give one file per <LinesPerFile> rows, named so that two runs
    on the same minute (or even the same second) cannot collide. Each
    rotated file gets a fresh header by default (WriteHeaderOnRotation).
    """

    DEFAULTS = {
        'OutputDir': 'data/onair/csv',
        'FilenameTemplate': 'csv_out_{timestamp}_pid{pid}',
        'LinesPerFile': '1000',
        'WriteHeaderOnRotation': 'true',
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

        self.pid = os.getpid()
        self.headers_built = False
        self.headers_written_to_current_file = False
        self.lines_current = 0
        self.current_buffer = []
        self.file_name = None  # set lazily on first render_reasoning()

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

    def _make_filename(self):
        # Microsecond precision so a rapid rotation cycle (e.g. small
        # LinesPerFile in tests, or a high-throughput scenario) cannot
        # produce two files with the same name within the same process.
        ts = datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
        name = self.filename_template.format(timestamp=ts, pid=self.pid) + '.csv'
        return os.path.join(self.output_dir, name)

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

        if not self.headers_written_to_current_file:
            with open(self.file_name, 'a', newline='') as f:
                csv.writer(f).writerow(self.headers)
            self.headers_written_to_current_file = True

        with open(self.file_name, 'a', newline='') as f:
            csv.writer(f).writerow(self.current_buffer)
        self.current_buffer = []
        self.lines_current += 1

        if self.lines_per_file != 0 and self.lines_current >= self.lines_per_file:
            self.file_name = self._make_filename()
            self.lines_current = 0
            # If WriteHeaderOnRotation is true, leave headers_written False so
            # the next render_reasoning() emits a header. Otherwise, suppress.
            self.headers_written_to_current_file = not self.write_header_on_rotation
