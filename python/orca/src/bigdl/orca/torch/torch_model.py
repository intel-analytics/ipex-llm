#
# Copyright 2018 Analytics Zoo Authors.
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

import torch

from bigdl.nn.layer import Layer
from bigdl.util.common import JTensor
from zoo.common.utils import callZooFunc
from pyspark.serializers import CloudPickleSerializer

if sys.version >= '3':
    long = int
    unicode = str


class TorchModel(Layer):
    """
    TorchModel wraps a PyTorch model as a single layer, thus the PyTorch model can be used for
    distributed inference or training.
    The implement of TorchModel is different from TorchNet, this TorchModel is running running
    pytorch model in an embedding Cpython interpreter, while TorchNet transfer the pytorch model
    to TorchScript and run with libtorch.
    """

    def __init__(self, module_bytes, weights, bigdl_type="float"):
        weights = JTensor.from_ndarray(weights)
        self.value = callZooFunc(
            bigdl_type, self.jvm_class_constructor(), module_bytes, weights)
        self.bigdl_type = bigdl_type

    @staticmethod
    def from_pytorch(model):
        """
        Create a TorchNet directly from PyTorch model, e.g. model in torchvision.models.
        :param model: a PyTorch model
        """
        weights = []
        for param in model.parameters():
            weights.append(param.view(-1))
        flatten_weight = torch.nn.utils.parameters_to_vector(weights).data.numpy()
        bys = CloudPickleSerializer.dumps(CloudPickleSerializer, model)
        net = TorchModel(bys, flatten_weight)
        return net
