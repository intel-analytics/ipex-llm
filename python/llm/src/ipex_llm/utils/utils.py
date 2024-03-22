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
from ipex_llm.utils.common import invalidInputError, invalidOperationError


def get_shared_lib_info(lib_base_name: str):
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        lib_ext = ".so"
    elif sys.platform == "win32":
        lib_ext = ".dll"
    else:
        invalidInputError(False, "Unsupported platform.")

    # Construct the paths to the possible shared library names (python/llm/src/bigdl/llm/libs)
    _base_path = pathlib.Path(__file__).parent.parent.resolve()
    _base_path = _base_path / 'libs'
    # Searching for the library in the current directory under the name "lib{lib_base_name}"
    # (default name for llmcpp) and "{lib_base_name}" (default name for this repo)
    _lib_paths = [
        _base_path / f"lib{lib_base_name}-api{lib_ext}",
        _base_path / f"{lib_base_name}-api{lib_ext}",
    ]

    return _base_path, _lib_paths
