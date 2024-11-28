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

import os
import sys
import ctypes
import pathlib
from ipex_llm.utils.common import invalidInputError


def get_shared_lib_info(lib_base_name: str):
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        lib_ext = ".so"
    elif sys.platform == "win32":
        lib_ext = ".dll"
    else:
        invalidInputError(False, "Unsupported platform.")

    # Construct the paths to the possible shared library names
    import importlib
    module = importlib.import_module("bigdl-core-npu")
    _base_path = pathlib.Path(module.__file__).parent.resolve()

    lib_path = os.path.join(_base_path, lib_base_name + lib_ext)

    return _base_path, lib_path


_, _lib_path = get_shared_lib_info("npu_llm")

# Load the library
_lib = ctypes.cdll.LoadLibrary(_lib_path)

_lib.load_model_from_file.argtypes = [ctypes.c_char_p]
_lib.load_model_from_file.restype = ctypes.c_void_p

_lib.run_prefill.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.run_prefill.restype = ctypes.POINTER(ctypes.c_float)

_lib.run_decode.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.run_decode.restype = ctypes.POINTER(ctypes.c_float)

_lib.llm_sample_token.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_bool, ctypes.c_int]
_lib.llm_sample_token.restype = ctypes.c_int

_lib.reset.argtypes = [ctypes.c_void_p]
_lib.reset.restype = None


def load_model_from_file(model_dir: str):
    return _lib.load_model_from_file(model_dir.encode('utf-8'))


def run_prefill(model_ptr, input_ids, vocab_size):
    input_ptr = (ctypes.c_int32 * len(input_ids))(*input_ids)
    input_len = len(input_ids)
    plogits = _lib.run_prefill(model_ptr, input_ptr, input_len)
    new_token = _lib.llm_sample_token(plogits, True, vocab_size)
    return new_token


def run_decode(model_ptr, input_id, vocab_size):
    plogits = _lib.run_decode(model_ptr, input_id)
    new_token = _lib.llm_sample_token(plogits, True, vocab_size)
    return new_token


def reset(model_ptr):
    _lib.reset(model_ptr)
