# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
SBN_Adapter class

Receives messages from SBN, serves as a data source for sim.py
"""

import threading
import time
import datetime
import os
import json

from onair.data_handling.on_air_data_source import OnAirDataSource
from onair.data_handling.on_air_data_source import ConfigKeyError
from ctypes import *
import sbn_python_client as sbn
import message_headers as msg_hdr

from onair.data_handling.parser_util import *

def _ctypes_to_python(obj):
    """Recursively convert a ctypes object to Python-native types.
    Array of structs, array of arrays, and nested structs are all flattened
    to nested Python lists of scalars (int/float/bytes)."""
    if isinstance(obj, Array):
        return [_ctypes_to_python(item) for item in obj]
    elif isinstance(obj, Structure):
        return [_ctypes_to_python(getattr(obj, f)) for f, _ in obj._fields_]
    else:
        return obj  # already a Python scalar (int, float, bytes)

# CFE_TBL name/path fields (LastFileLoaded etc.) are strings (c_char[]); the IF/
# XGB models can't consume them, and writing them raw re-introduces the
# text<->numeric schema drift that got them ExcludeColumns-suppressed. Instead we
# emit two *numeric* derived columns per field: a frame-to-frame "changed" flag
# (a table load/update event just occurred) and a monotonic count of those
# events. This is the AINOS3-30 signal for the table-load DEAD classes
# (DE-0003.03/.08/.09). Order here MUST match the tail of the tlm.json `order`
# list (and thus all_headers), since the derived columns are appended to both.
#   (raw_header, changed_col, count_col)
_DERIVED_TBL_FIELDS = [
    ("CFE_TBL.LastFileLoaded",   "CFE_TBL.FileLoadChanged",    "CFE_TBL.FileLoadCount"),
    ("CFE_TBL.LastUpdatedTable", "CFE_TBL.TableUpdateChanged", "CFE_TBL.TableUpdateCount"),
    ("CFE_TBL.LastTableLoaded",  "CFE_TBL.TableLoadChanged",   "CFE_TBL.TableLoadCount"),
]


def _tbl_change_detect(cur_value, prev_value, prev_count):
    """Frame-to-frame change detection for a CFE_TBL name/path string field.

    Returns (changed, count, normalized_current):
      changed            : 1 when a non-first observation differs from the
                           previous value (a load/update event), else 0.
      count              : monotonic event count (prev_count, +1 on a change).
      normalized_current : current value coerced to str ("" for the [0] init
                           sentinel), to be stored as the next prev_value.

    The first observation (prev_value is None) never counts as an event — the
    boot-time table load is nominal, not an anomaly. These fields fire on a
    table/file NAME change and complement CFE_TBL.LastUpdateTime* (which also
    moves on a same-name content reload)."""
    cur = cur_value if isinstance(cur_value, str) else ""
    if prev_value is None:
        return 0, prev_count, cur
    if cur != prev_value:
        return 1, prev_count + 1, cur
    return 0, prev_count, cur

# Note: The double buffer does not clear between switching. If fresh data doesn't come in, stale data is returned (delayed by 1 frame)

class DataSource(OnAirDataSource):

    def __init__(self, data_file, meta_file, ss_breakdown = False):
        super().__init__(data_file, meta_file, ss_breakdown);

        self.new_data_lock = threading.Lock()
        self.new_data = False
        self.double_buffer_read_index = 0
        # Per-field state for CFE_TBL name-change detection (see
        # _DERIVED_TBL_FIELDS / _update_tbl_derived). Initialized before
        # connect() launches the listener thread that reads them.
        self._tbl_prev = {}
        self._tbl_count = {}
        self.connect()

    def connect(self):
        """Establish connection to SBN and launch listener thread."""
        time.sleep(2)
        os.chdir("cf")
        sbn.sbn_load_and_init()
        os.chdir("../")
        print("SBN_Adapter Running")

        # Launch thread to listen for messages
        self.listener_thread = threading.Thread(target=self.message_listener_thread)
        self.listener_thread.start()

        # subscribe to message IDs
        for msgID in self.msgID_lookup_table.keys():
            sbn.subscribe(msgID)

    def gather_field_names(self, field_name, field_type):

        # recursively find field names in DFS manner
        def gather_field_names_helper(field_name:str, field_type, field_names:list):
            if "message_headers" in str(field_type) and hasattr(field_type, "_fields_"):
                for sub_field_name, sub_field_type in field_type._fields_:
                    gather_field_names_helper(field_name + "." + sub_field_name, sub_field_type,field_names)
            else:
                field_names.append(field_name)

        field_names = []
        gather_field_names_helper(field_name, field_type, field_names)
        return field_names

    def parse_meta_data_file(self, meta_data_file, ss_breakdown):
        self.msgID_lookup_table = {}
        self.currentData = []

        # pull out message ids
        file = open(meta_data_file, 'rb')
        file_str = file.read()

        meta_config = json.loads(file_str)
        file.close()

        if 'channels' not in meta_config.keys():
            raise ConfigKeyError(f'Config file: \'{meta_data_file}\' ' \
                                  'missing required key \'channels\'')

        # Copy message ID table from .json, convert string hex to ints for ID
        for key in meta_config['channels']:
            self.msgID_lookup_table[int(key, 16)] = meta_config['channels'][key]

        # Use eval() to convert class name from .json to match with message_headers.py
        for key in self.msgID_lookup_table:
            msg_struct_name = self.msgID_lookup_table[key][1]
            self.msgID_lookup_table[key][1] = eval("msg_hdr." + msg_struct_name)

        # populate headers and reserve space for data
        for x in range(0,2):
            self.currentData.append({'headers':[], 'data':[]})

            for msgID in self.msgID_lookup_table.keys():
                app_name, data_struct = self.msgID_lookup_table[msgID]
                struct_name = data_struct.__name__
                # Skip the header, walk through the stuct
                for field_name, field_type in data_struct._fields_[1:]:
                    field_names = self.gather_field_names(app_name + "." + field_name, field_type)
                    for field_name in field_names:
                        self.currentData[x]['headers'].append(field_name)
                        self.currentData[x]['data'].append([0]) #initialize all the data arrays with zero

            # Derived CFE_TBL name-change features (numeric). Appended AFTER all
            # struct fields, and ONLY when the raw source field is present in
            # this schema — so a schema without CFE_TBL (or without these entries
            # in its `order` list) doesn't gain currentData columns that
            # all_headers lacks. The tlm.json `order` tail must list exactly the
            # derived columns whose raw field is subscribed. get_current_data()
            # populates them.
            for raw_h, changed_h, count_h in _DERIVED_TBL_FIELDS:
                if raw_h not in self.currentData[x]['headers']:
                    continue
                # Init to scalar 0 (NOT the [0] array sentinel used for struct
                # fields): these are scalars, and a frame emitted from a buffer
                # that hasn't yet processed a CFE_TBL message must still read a
                # numeric 0, never the literal "[0]" string in the CSV.
                self.currentData[x]['headers'].append(changed_h)
                self.currentData[x]['data'].append(0)
                self.currentData[x]['headers'].append(count_h)
                self.currentData[x]['data'].append(0)
        print("Current Data Headers: {}.".format(self.currentData[0]["headers"]))
        return extract_meta_data_handle_ss_breakdown(meta_data_file, ss_breakdown)

    def process_data_file(self, data_file):
        print("SBN Adapter ignoring data file (telemetry should be live)")

    def get_vehicle_metadata(self):
        return self.all_headers, self.binning_configs['test_assignments']

    def get_next(self):
        """Provides the latest data from SBN in a dictionary of lists structure.
        Returned data is safe to use until the next get_next call.
        Blocks until new data is available."""

        data_available = False

        while not data_available:
            with self.new_data_lock:
                data_available = self.has_data()

            if not data_available:
                time.sleep(0.01)

        read_index = 0
        with self.new_data_lock:
            self.new_data = False
            self.double_buffer_read_index = (self.double_buffer_read_index + 1) % 2
            read_index = self.double_buffer_read_index

        return self.currentData[read_index]['data']

    def has_more(self):
        """Returns true if the adapter has more data.
           For now always true: connection should be live as long as cFS is running.
           TODO: allow to detect if cFS/the connection has died"""
        return True

    def message_listener_thread(self):
        """Thread to listen for incoming messages from SBN"""
        # Track unknown MIDs we've already warned about so the log isn't spammed
        # under a fuzz attack (EX-0009.01-style) that delivers many unknown StreamIds.
        unknown_mids_seen = set()

        while(True):
            generic_recv_msg_p = POINTER(sbn.sbn_data_generic_t)()
            sbn.recv_msg(generic_recv_msg_p)

            msgID = generic_recv_msg_p.contents.TlmHeader.Primary.StreamId
            try:
                app_name, data_struct = self.msgID_lookup_table[msgID]
            except KeyError:
                # Unknown / malformed StreamId — common under fuzz attacks or
                # FSW init handshake. Skip the packet rather than letting the
                # KeyError propagate and kill the listener thread (which would
                # back up SBN with "pipe overflow" floods and break detection).
                if msgID not in unknown_mids_seen:
                    print(f"[sbn_adapter] WARNING: unknown StreamId 0x{msgID:04X}; skipping (logged once per MID)")
                    unknown_mids_seen.add(msgID)
                continue

            recv_msg_p = POINTER(data_struct)()
            recv_msg_p.contents = generic_recv_msg_p.contents
            recv_msg = recv_msg_p.contents

            # prints out the data from the message to the terminal
            print(", ".join([field_name + ": " + str(getattr(recv_msg, field_name)) for field_name, field_type in recv_msg._fields_[1:]]))

            # TODO: Lock needed here?
            self.get_current_data(recv_msg, data_struct, app_name)

    def get_current_data(self, recv_msg, data_struct, app_name):
        # TODO: Lock needed here?
        current_buffer = self.currentData[(self.double_buffer_read_index + 1) %2]

        # Skip the header, walk through the stuct
        for field_name, field_type in recv_msg._fields_[1:]:
            field_names = self.gather_field_names(field_name, field_type)

            for name in field_names:
                idx = current_buffer['headers'].index(app_name + "." + name)
                # Pull the data out of the message by walking down the nested types
                current_object = recv_msg
                for sub_type in name.split('.'):
                    current_object = getattr(current_object, sub_type)
                if isinstance(current_object, (Array, Structure)):
                    data = _ctypes_to_python(current_object)
                elif isinstance(current_object, bytes):
                    # ctypes c_char * N fields surface as bytes here (not Array).
                    # Decode them once, stripping nul-padding, so downstream
                    # consumers see clean text instead of "b'X'" reprs.
                    data = current_object.rstrip(b'\x00').decode('utf-8', errors='replace')
                else:
                    data = str(current_object)
                current_buffer['data'][idx] = data

        # Once the raw CFE_TBL name fields for this frame are populated, derive
        # their numeric change-detection columns (AINOS3-30).
        if app_name == "CFE_TBL":
            self._update_tbl_derived(current_buffer)

        with self.new_data_lock:
            self.new_data = True

    def _update_tbl_derived(self, current_buffer):
        """Write the numeric CFE_TBL name-change features into current_buffer.

        Reads each raw name/path field (already populated for this frame),
        computes its frame-to-frame change flag + monotonic event count via
        _tbl_change_detect, and stores them in the paired derived columns.
        State (previous value + count) is kept per-field on the instance so it
        persists across frames. A missing header is skipped defensively."""
        headers = current_buffer['headers']
        data = current_buffer['data']
        for raw_h, changed_h, count_h in _DERIVED_TBL_FIELDS:
            try:
                raw_val = data[headers.index(raw_h)]
            except ValueError:
                continue
            changed, count, norm = _tbl_change_detect(
                raw_val, self._tbl_prev.get(raw_h), self._tbl_count.get(raw_h, 0))
            self._tbl_prev[raw_h] = norm
            self._tbl_count[raw_h] = count
            try:
                data[headers.index(changed_h)] = changed
                data[headers.index(count_h)] = count
            except ValueError:
                pass

    def has_data(self):
        return self.new_data
