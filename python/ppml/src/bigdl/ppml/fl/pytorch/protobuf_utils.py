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
from bigdl.ppml.fl.pytorch.generated.fl_base_pb2 import FloatTensor, TensorMap

def ndarray_map_to_tensor_map(array_map: dict):
    tensor_map = {}
    for (k, v) in array_map.items():
        if not isinstance(v, np.ndarray):
            raise Exception("ndarray map element should be Numpy ndarray")
        tensor_map[k] = FloatTensor(tensor=v.flatten().tolist(),shape=v.shape)
    return TensorMap(tensorMap=tensor_map)


def tensor_map_to_ndarray_map(tensor_map: TensorMap):
    ndarray_map = {}
    for (k, v) in tensor_map.items():
        if not isinstance(v, FloatTensor):
            raise Exception("tensor map element should be protobuf type FloatTensor")
        dtype = "long" if k == 'target' else "float32"
        ndarray_map[k] = np.array(v.tensor, dtype=dtype).reshape(v.shape)
    return ndarray_map