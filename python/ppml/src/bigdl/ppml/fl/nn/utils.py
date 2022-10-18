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

import logging
import pickle
import stat
import numpy as np

from bigdl.ppml.fl.nn.generated.nn_service_pb2 import *
from bigdl.ppml.fl.nn.generated.fl_base_pb2 import FloatTensor, TensorMap


class ClassAndArgsWrapper(object):
    def __init__(self, cls, args) -> None:
        self.cls = cls
        self.args = args

    def to_protobuf(self):
        cls = pickle.dumps(self.cls)
        args = pickle.dumps(self.args)
        return ClassAndArgs(cls=cls, args=args)



import numpy as np
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.ppml.fl.nn.generated.fl_base_pb2 import FloatTensor, TensorMap

def ndarray_map_to_tensor_map(array_map: dict):
    tensor_map = {}
    for (k, v) in array_map.items():
        if not isinstance(v, np.ndarray):
            invalidInputError(False,
                              "ndarray map element should be Numpy ndarray")
        tensor_map[k] = FloatTensor(tensor=v.flatten().tolist(), shape=v.shape, dtype=str(v.dtype))
    return TensorMap(tensorMap=tensor_map)


def tensor_map_to_ndarray_map(tensor_map: TensorMap):
    ndarray_map = {}
    for (k, v) in tensor_map.items():
        if not isinstance(v, FloatTensor):
            invalidInputError(False,
                              "tensor map element should be protobuf type FloatTensor")
        dtype = "float32" if v.dtype is None else v.dtype
        ndarray_map[k] = np.array(v.tensor, dtype=dtype).reshape(v.shape)
    return ndarray_map

def file_chunk_generate(file_path):
    CHUNK_SIZE = 1 * 1024 * 1024
    logging.debug("Splitting model to file chunks")
    with open(file_path, 'rb') as f:
        while True:
            piece = f.read(CHUNK_SIZE);
            if not piece:
                return
            yield ByteChunk(buffer=piece)

def print_file_size_in_dir(path='.'):
    import os
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                print(entry, entry.stat().st_size)
            elif entry.is_dir():
                print_file_size_in_dir(entry.path)
