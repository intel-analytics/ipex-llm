#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import ctypes
import base64
import os

def generate_tdx_quote(user_report_data):
    # Define the uuid data structure
    TDX_UUID_SIZE = 16
    class TdxUuid(ctypes.Structure):
        _fields_ = [("d", ctypes.c_uint8 * TDX_UUID_SIZE)]

    # Define the report data structure
    TDX_REPORT_DATA_SIZE = 64
    class TdxReportData(ctypes.Structure):
        _fields_ = [("d", ctypes.c_uint8 * TDX_REPORT_DATA_SIZE)]

    # Define the report structure
    TDX_REPORT_SIZE = 1024
    class TdxReport(ctypes.Structure):
        _fields_ = [("d", ctypes.c_uint8 * TDX_REPORT_SIZE)]

    # Load the library
    tdx_attest = ctypes.cdll.LoadLibrary("/usr/lib/x86_64-linux-gnu/libtdx_attest.so")

    # Set the argument and return types for the function
    tdx_attest.tdx_att_get_report.argtypes = [ctypes.POINTER(TdxReportData), ctypes.POINTER(TdxReport)]
    tdx_attest.tdx_att_get_report.restype = ctypes.c_uint16

    tdx_attest.tdx_att_get_quote.argtypes = [ctypes.POINTER(TdxReportData), ctypes.POINTER(TdxUuid), ctypes.c_uint32, ctypes.POINTER(TdxUuid), ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    tdx_attest.tdx_att_get_quote.restype = ctypes.c_uint16


    # Call the function and check the return code
    byte_array_data = bytearray(user_report_data.ljust(64)[:64], "utf-8").replace(b' ', b'\x00')
    report_data = TdxReportData()
    report_data.d = (ctypes.c_uint8 * 64).from_buffer(byte_array_data)
    report = TdxReport()
    result = tdx_attest.tdx_att_get_report(ctypes.byref(report_data), ctypes.byref(report))
    if result != 0:
        print("Error: " + hex(result))

    att_key_id_list = None
    list_size = 0
    att_key_id = TdxUuid()
    p_quote = ctypes.POINTER(ctypes.c_uint8)()
    quote_size = ctypes.c_uint32()
    flags = 0

    result = tdx_attest.tdx_att_get_quote(ctypes.byref(report_data), att_key_id_list, list_size, ctypes.byref(att_key_id), ctypes.byref(p_quote), ctypes.byref(quote_size), flags)

    if result != 0:
        print("Error: " + hex(result))
    else:
        quote = ctypes.string_at(p_quote, quote_size.value)
        return quote

def generate_gramine_quote(user_report_data):
    USER_REPORT_PATH = "/dev/attestation/user_report_data"
    QUOTE_PATH = "/dev/attestation/quote"
    if not os.path.isfile(USER_REPORT_PATH):
        print(f"File {USER_REPORT_PATH} not found.")
        return ""
    if not os.path.isfile(QUOTE_PATH):
        print(f"File {QUOTE_PATH} not found.")
        return ""
    with open(USER_REPORT_PATH, 'w') as out:
        out.write(user_report_data)
    with open(QUOTE_PATH, "rb") as f:
        quote = f.read()
    return quote

