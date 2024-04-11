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

from vllm.entrypoints import api_server as vllm_api_server

import sys

def run_vllm_api_server():
    if len(sys.argv) > 1:
        sys.argv = [sys.argv[0]] + sys.argv[1:]
    else:
        sys.argv = [sys.argv[0]]

    vllm_api_server.main()

if __name__ == "__main__":
    run_vllm_api_server()