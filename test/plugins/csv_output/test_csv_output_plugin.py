# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test CSV Output Plugin Functionality """
import os
import shutil
import tempfile
from copy import copy
from unittest.mock import MagicMock

import pytest

from plugins.csv_output import csv_output_plugin
from plugins.csv_output.csv_output_plugin import Plugin as CSV_Output


@pytest.fixture
def isolated_outdir(monkeypatch, tmp_path):
    """Force OutputDir into a tmp dir so tests never write into the repo."""
    monkeypatch.setattr(
        CSV_Output, 'DEFAULTS',
        {**CSV_Output.DEFAULTS, 'OutputDir': str(tmp_path)},
    )
    # Prevent _load_config from picking up a real ini on this host.
    monkeypatch.setenv('ONAIR_INI_FILE', '/nonexistent/onair.ini')
    return tmp_path


def test_init_initializes_expected_default_variables(isolated_outdir):
    arg_name = MagicMock()
    arg_headers = [MagicMock(), MagicMock()]

    csv_out = CSV_Output(arg_name, arg_headers)

    assert csv_out.component_name == arg_name
    assert csv_out.headers == arg_headers
    assert csv_out.lines_per_file == int(CSV_Output.DEFAULTS['LinesPerFile'])
    assert csv_out.lines_current == 0
    assert csv_out.current_buffer == []
    assert csv_out.headers_built is False
    assert csv_out.headers_written_to_current_file is False
    assert csv_out.file_name is None
    assert csv_out.write_header_on_rotation is True
    assert csv_out.output_dir == str(isolated_outdir)


def test_update_adds_plugins_to_headers_only_once(isolated_outdir):
    arg_headers = ['header1', 'header2']
    csv_out = CSV_Output(MagicMock(), arg_headers)

    high_level_data = {
        'layer1': {'plugin1': []},
        'layer2': {'plugin2': [], 'plugin3': []},
    }
    initial_headers = copy(csv_out.headers)
    expected_headers = initial_headers + ['plugin1', 'plugin2', 'plugin3']

    csv_out.update([], high_level_data)
    assert csv_out.headers == expected_headers
    assert csv_out.headers_built is True

    # second update must not re-append
    csv_out.update([], high_level_data)
    assert csv_out.headers == expected_headers


def test_update_does_not_add_headers_when_no_plugins(isolated_outdir):
    arg_headers = ['header1', 'header2']
    csv_out = CSV_Output(MagicMock(), arg_headers)

    high_level_data = {'layer1': {}, 'layer2': {}}
    initial_headers = copy(csv_out.headers)

    csv_out.update([], high_level_data)
    assert csv_out.headers == initial_headers


def test_update_does_not_add_headers_when_no_layers(isolated_outdir):
    arg_headers = ['header1', 'header2']
    csv_out = CSV_Output(MagicMock(), arg_headers)
    initial_headers = copy(csv_out.headers)

    csv_out.update([], {})
    assert csv_out.headers == initial_headers


def test_update_leaves_buffer_empty_when_given_no_data(isolated_outdir):
    csv_out = CSV_Output(MagicMock(), [MagicMock(), MagicMock()])

    csv_out.update(low_level_data=[], high_level_data={})
    assert csv_out.current_buffer == []


def test_update_fills_buffer_with_low_level_data(isolated_outdir):
    csv_out = CSV_Output(MagicMock(), [MagicMock(), MagicMock()])

    low_level_data = [MagicMock() for _ in range(10)]
    csv_out.update(low_level_data, {})
    assert csv_out.current_buffer == [str(item) for item in low_level_data]


def test_update_fills_buffer_with_high_level_data(isolated_outdir):
    csv_out = CSV_Output(MagicMock(), [MagicMock(), MagicMock()])

    example = {
        'vehicle_rep': {'plugin_1': ['1', '2', '3']},
        'learning_system': {'plugin_2': ['a', 'b', 'c'], 'plugin_3': ['x', 'y', 'z']},
        'planning_system': {},
    }
    expected_buffer = ['1', '2', '3', 'a', 'b', 'c', 'x', 'y', 'z']

    csv_out.update(low_level_data=[], high_level_data=example)
    assert csv_out.current_buffer == expected_buffer


def test_render_reasoning_writes_header_then_data_to_first_file(isolated_outdir):
    csv_out = CSV_Output(MagicMock(), ['h1', 'h2'])
    csv_out.update(['v1', 'v2'], {})

    csv_out.render_reasoning()

    assert csv_out.file_name is not None
    assert os.path.dirname(csv_out.file_name) == str(isolated_outdir)
    with open(csv_out.file_name) as f:
        contents = f.read().splitlines()
    assert contents == ['h1,h2', 'v1,v2']


def _list_csv_files(outdir):
    """Filter to .csv files; the writer also drops .meta.json sidecars."""
    return sorted(f for f in os.listdir(outdir) if f.endswith('.csv'))


def test_render_reasoning_rotates_after_lines_per_file(isolated_outdir):
    csv_out = CSV_Output(MagicMock(), ['h1', 'h2'])
    csv_out.lines_per_file = 1  # rotate after every data row

    csv_out.update(['a', 'b'], {}); csv_out.render_reasoning()
    csv_out.update(['c', 'd'], {}); csv_out.render_reasoning()
    csv_out.update(['e', 'f'], {}); csv_out.render_reasoning()

    files = _list_csv_files(isolated_outdir)
    assert len(files) == 3
    contents = [open(os.path.join(isolated_outdir, f)).read().splitlines() for f in files]
    # Each rotated file gets its own header by default.
    assert contents == [['h1,h2', 'a,b'], ['h1,h2', 'c,d'], ['h1,h2', 'e,f']]


def test_render_reasoning_suppresses_header_when_configured(isolated_outdir):
    csv_out = CSV_Output(MagicMock(), ['h1', 'h2'])
    csv_out.lines_per_file = 1
    csv_out.write_header_on_rotation = False

    csv_out.update(['a', 'b'], {}); csv_out.render_reasoning()
    csv_out.update(['c', 'd'], {}); csv_out.render_reasoning()
    csv_out.update(['e', 'f'], {}); csv_out.render_reasoning()

    files = _list_csv_files(isolated_outdir)
    contents = [open(os.path.join(isolated_outdir, f)).read().splitlines() for f in files]
    # First file has the header; rotated files do not.
    assert contents == [['h1,h2', 'a,b'], ['c,d'], ['e,f']]


def test_exclude_columns_drops_specified_headers_and_aligned_data(isolated_outdir):
    csv_out = CSV_Output(MagicMock(), ['keep1', 'drop_me', 'keep2', 'also_drop'])
    csv_out.exclude_columns = {'drop_me', 'also_drop'}

    csv_out.update(['k1v', 'dv', 'k2v', 'adv'], {})
    csv_out.render_reasoning()

    with open(csv_out.file_name) as f:
        contents = f.read().splitlines()
    # Header + one data row; both columns at indices 1 and 3 are dropped.
    assert contents == ['keep1,keep2', 'k1v,k2v']


def test_exclude_columns_unknown_name_is_a_noop(isolated_outdir):
    csv_out = CSV_Output(MagicMock(), ['h1', 'h2'])
    csv_out.exclude_columns = {'not_a_real_header'}

    csv_out.update(['a', 'b'], {})
    csv_out.render_reasoning()

    with open(csv_out.file_name) as f:
        contents = f.read().splitlines()
    assert contents == ['h1,h2', 'a,b']


def test_meta_sidecar_written_at_file_creation(isolated_outdir):
    import json
    csv_out = CSV_Output(MagicMock(), ['h1', 'h2', 'drop'])
    csv_out.exclude_columns = {'drop'}

    csv_out.update(['v1', 'v2', 'v3'], {})
    csv_out.render_reasoning()

    sidecar = csv_out.file_name[:-4] + '.meta.json'
    assert os.path.exists(sidecar), 'meta sidecar should be written at file creation'
    meta = json.loads(open(sidecar).read())
    assert meta['csv_basename'] == os.path.basename(csv_out.file_name)
    assert meta['pid'] == os.getpid()
    assert meta['kept_columns'] == ['h1', 'h2']
    assert meta['kept_columns_count'] == 2
    assert meta['excluded_columns'] == ['drop']
    assert 'container' in meta
    assert 'plugin_version' in meta
    # schema_sha256 may be None when no [FILES] section in active ini


def test_ini_overrides_defaults(monkeypatch, tmp_path):
    ini = tmp_path / 'test.ini'
    ini.write_text(
        '[CSV_OUTPUT]\n'
        f'OutputDir = {tmp_path}\n'
        'LinesPerFile = 7\n'
        'WriteHeaderOnRotation = false\n'
        'ExcludeColumns = foo, bar ,baz\n'
    )
    monkeypatch.setenv('ONAIR_INI_FILE', str(ini))

    csv_out = CSV_Output(MagicMock(), ['h'])
    assert csv_out.output_dir == str(tmp_path)
    assert csv_out.lines_per_file == 7
    assert csv_out.write_header_on_rotation is False
    assert csv_out.exclude_columns == {'foo', 'bar', 'baz'}
