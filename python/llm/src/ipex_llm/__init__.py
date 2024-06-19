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

# This would makes sure Python is aware there is more than one sub-package within bigdl,
# physically located elsewhere.
# Otherwise there would be module not found error in non-pip's setting as Python would
# only search the first bigdl package and end up finding only one sub-package.

from .convert_model import llm_convert
from .optimize import optimize_model
import os
from .llm_patching import llm_patch, llm_unpatch
import sys
import types

# Default is True, set to False to disable auto importing Intel Extension for PyTorch.
USE_NPU = os.getenv("BIGDL_USE_NPU", 'False').lower() in ('true', '1', 't')
BIGDL_IMPORT_IPEX = os.getenv("BIGDL_IMPORT_IPEX", 'True').lower() in ('true', '1', 't')
BIGDL_IMPORT_IPEX = not USE_NPU and BIGDL_IMPORT_IPEX
if BIGDL_IMPORT_IPEX:
    # Import Intel Extension for PyTorch as ipex if XPU version is installed
    from .utils.ipex_importer import ipex_importer
    # Avoid duplicate import
    if ipex_importer.get_ipex_version() is None:
        ipex_importer.import_ipex()

# Default is true, set to true to auto patching bigdl-llm to ipex_llm.
BIGDL_COMPATIBLE_MODE = os.getenv("BIGDL_COMPATIBLE_MODE", 'True').lower() in ('true', '1', 't')
if BIGDL_COMPATIBLE_MODE:
    # Make users' application with previous bigdl-llm could run easily through this patch
    # Avoid ModuleNotFoundError of 'bigdl', map 'bigdl' to a dummy module
    sys.modules['bigdl'] = types.ModuleType('_ipex_llm_dummy')
    # Map 'bigdl.llm' to 'ipex_llm'
    sys.modules['bigdl.llm'] = sys.modules['ipex_llm']
