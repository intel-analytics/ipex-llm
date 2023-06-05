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
from bigdl.llm.utils.common import invalidInputError, invalidOperationError


def get_avx_flags():
    avx = ""
    if sys.platform != "win32":
        import subprocess
        msg = subprocess.check_output(["lscpu"]).decode("utf-8")
        if "avx512_vnni" in msg:
            avx = "_avx512"
        elif "avx2" in msg:
            avx = "_avx2"
        else:
            invalidOperationError(False, "Unsupported CPUFLAGS.")
    return avx
