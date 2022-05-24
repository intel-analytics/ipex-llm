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


import torch
from torch import nn
import pickle
from bigdl.ppml.fl.pytorch.generated.nn_service_pb2 import *

def set_one_like_parameter(model: nn.Module):
    for param in model.parameters():
        param.data = nn.parameter.Parameter(torch.ones_like(param))

class ClassAndArgsWrapper(object):
    def __init__(self, cls, args) -> None:
        self.cls = cls
        self.args = args

    def to_protobuf(self):
        cls = pickle.dumps(self.cls)
        args = pickle.dumps(self.args)
        return ClassAndArgs(cls=cls, args=args)
