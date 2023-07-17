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

import sys
import pathlib
from isa_checker import check_avx_vnni, check_avx2, check_avx512_vnni
from bigdl.llm.utils.common import invalidInputError, invalidOperationError


def get_avx_flags():
    avx = ""
    if sys.platform != "win32":
        if check_avx512_vnni():
            avx = "_avx512"
        elif check_avx2():
            avx = "_avx2"
        else:
            invalidOperationError(False, "Unsupported CPUFLAGS.")
    return avx


def get_shard_lib_info(lib_base_name: str):
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        lib_ext = ".so"
        vnni = ""
    elif sys.platform == "win32":
        lib_ext = ".dll"
        vnni = "_vnni" if check_avx_vnni() else ""
    else:
        invalidInputError(False, "Unsupported platform.")

    avx = get_avx_flags()

    # Construct the paths to the possible shared library names (python/llm/src/bigdl/llm/libs)
    _base_path = pathlib.Path(__file__).parent.parent.resolve()
    _base_path = _base_path / 'libs'
    # Searching for the library in the current directory under the name "lib{lib_base_name}"
    # (default name for llmcpp) and "{lib_base_name}" (default name for this repo)
    _lib_paths = [
        _base_path / f"lib{lib_base_name}{avx}{vnni}{lib_ext}",
        _base_path / f"{lib_base_name}{avx}{vnni}{lib_ext}",
    ]

    return _base_path, _lib_paths
