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
from bigdl.contrib.onnx import load


def load_onnx_resnet():
    restnet_path = "./resnet-50.onnx"
    restnet_tensor = np.random.random([10, 3, 224, 224])
    restnet = load(restnet_path)
    restnet_out = restnet.forward(restnet_tensor)
    return restnet_out


if __name__ == "__main__":
    load_onnx_resnet()
