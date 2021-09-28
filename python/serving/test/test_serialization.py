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


import numpy as np
import base64
from bigdl.serving.client import InputQueue, OutputQueue, http_json_to_ndarray
import os


resource_path = os.path.join(os.path.split(__file__)[0], "../resources")


class TestSerialization:

    def test_encode(self):
        input_api = InputQueue()
        b64 = input_api.data_to_b64(t1=np.array([1, 2]), t2=np.array([3, 4]))
        byte = base64.b64decode(b64)

    def test_http_response_to_ndarray(self):
        with open(os.path.join(resource_path, "http_response")) as f:
            data = f.read()
            arr = http_json_to_ndarray(data)
            assert isinstance(arr, np.ndarray)
            assert len(arr.shape) == 1
            assert arr.shape[0] == 128
