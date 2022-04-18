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
from ...inference.pytorch.utility import export as export_to_onnx
import torch


class PytorchOpenVINOModel(OpenVINOModel, AcceleratedLightningModule):
    def __init__(self, model: torch.nn.Module, input_sample=None, xml_path="model.xml"):
        AcceleratedLightningModule.__init__(self, model)
        export(model, input_sample, xml_path)
        OpenVINOModel.__init__(self, xml_path)

    def on_forward_start(self, inputs):
        inputs = self.tensors_to_numpy(inputs)
        return super().on_forward_start(inputs)

    def on_forward_end(self, outputs):
        outputs = self.numpy_to_tensors(outputs.values())
        return super().on_forward_end(outputs)


def export(model, input_sample=None, xml_path="model.xml"):
    '''
    Internal function to build a ortsess and bind to the lightningmodule.

    :param input_sample: torch.Tensor or a list for the model tracing.
    :param file_path: The path to save openvino model file.
    '''
    export_to_onnx(model, input_sample, 'model.onnx', dynamic_axes=False)
    convert_onnx_to_xml('model.onnx', xml_path)
