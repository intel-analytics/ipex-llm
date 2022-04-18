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

from functools import partial
from ..core.openvino_model import OpenVINOModel
from ..core.utility import convert_onnx_to_xml
from ...inference.pytorch.model import AcceleratedLightningModule
from .utility import export
import torch


class PytorchOpenVINOModel(OpenVINOModel, AcceleratedLightningModule):
    def __init__(self, model: torch.nn.Module =None, input_sample=None):
        AcceleratedLightningModule.__init__(self, model)
        if model is not None:
            export(model, input_sample, 'tmp.xml')
            OpenVINOModel.__init__(self, 'tmp.xml')
            os.remove('tmp.xml')

    def on_forward_start(self, inputs):
        if self.ie_network is None:
            raise RuntimeError(
                "Please create an instance by PytorchOpenVINOModel() or PytorchOpenVINOModel.load()"
                )
        inputs = self.tensors_to_numpy(inputs)
        return inputs

    def on_forward_end(self, outputs):
        outputs = self.numpy_to_tensors(outputs.values())
        return outputs
