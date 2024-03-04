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

# Default is false, set to true to auto importing Intel Extension for PyTorch.
BIGDL_IMPORT_IPEX = os.getenv("BIGDL_IMPORT_IPEX", 'True').lower() in ('true', '1', 't')
if BIGDL_IMPORT_IPEX:
    # Import Intel Extension for PyTorch as ipex if XPU version is installed
    from .utils.ipex_importer import ipex_importer
    ipex_importer.import_ipex()
