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

    # Construct the paths to the possible shared library names (python/llm/src/ipex-llm/llm/libs)
    _base_path = pathlib.Path(__file__).parent.parent.parent.resolve()
    _base_path = _base_path / 'libs'

    lib_path = os.path.join(_base_path, lib_base_name + lib_ext)

    return _base_path, lib_path

_, _lib_path = get_shared_lib_info("pipeline")

# Load the library
_lib = ctypes.cdll.LoadLibrary(_lib_path)

_lib.InitLLMPipeline.argtypes = [ctypes.c_int] * 5 + [ctypes.c_char_p] * 5
_lib.InitLLMPipeline.restype = ctypes.c_int

_lib.generate_serve.argtypes = [ctypes.c_int] * 4 + [ctypes.c_bool] + [ctypes.c_int]
_lib.generate_serve.restype = ctypes.c_int


def InitLLMPipeline(kv_len: int, num_head: int, head_dim: int, num_layers: int, vocab_size: int,
                    model_weight_dir: str, model_name: str,
                    first_blob_name: str, last_blob_name: str, rest_blob_name: str):
    return _lib.InitLLMPipeline(kv_len, num_head, head_dim, num_layers, vocab_size,
                                model_weight_dir.encode('utf-8'), model_name.encode('utf-8'),
                                first_blob_name.encode('utf-8'), last_blob_name.encode('utf-8'),
                                rest_blob_name.encode('utf-8'))


def generate_serve(kv_len: int, num_head: int, head_dim: int, num_layers: int,
                   transpose_value_cache: bool, param_n_output: int):
    _lib.generate_serve(kv_len, num_head, head_dim, num_layers,
                        transpose_value_cache, param_n_output)
