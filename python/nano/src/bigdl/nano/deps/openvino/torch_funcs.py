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
from .base_openvino import BaseOpenVINOInference, convert_onnx_to_xml
from ..inference.pytorch_base_inference import PytorchBaseInference
from ..inference.pytorch_base_inference import export as export_to_onnx
import torch


class PytorchOpenVINOInference(BaseOpenVINOInference, PytorchBaseInference):
    def __init__(self, ie_network=None):
        BaseOpenVINOInference.__init__(self, ie_network)
        PytorchBaseInference.__init__(self)

    @staticmethod
    def from_torch(model: torch.nn.Module, input_sample=None, xml_path="model.xml"):
        export(model, input_sample, xml_path)
        openvino_inference = PytorchOpenVINOInference()
        openvino_inference.read_network(model=xml_path)
        return openvino_inference

    def attach(self, model):
        if not getattr(model, "_torch_train", None):
            model._torch_train = model.train
        model._torch_forward = model.forward

        model.forward = partial(self.forward, model)
        model.train = partial(self.train, model)

    def detach(self, model):
        model.forward = model._torch_forward

    def on_train_start(self, model, mode):
        self.detach(model)
        if mode:
            self.reset(model)


def export(model, input_sample=None, xml_path="model.xml"):
    '''
    Internal function to build a ortsess and bind to the lightningmodule.

    :param input_sample: torch.Tensor or a list for the model tracing.
    :param file_path: The path to save openvino model file.
    '''
    export_to_onnx(model, input_sample, 'model.onnx', dynamic_axes=False)
    convert_onnx_to_xml('model.onnx', xml_path)


def eval_openvino(model, input_sample=None, xml_path="model.xml"):
    '''
    This method change the `forward` method to an openvino backed forwarding.

    >>> model.eval_openvino(quantize=True/False)
    >>> pred = model(x)  # openvino forwarding
    >>> model.exit_openvino()

    :param input_sample: (optional) a torch dataloader, torch.Tensor or a
           list of them for the model tracing.
    :param file_path: (optional) The path to save openvino model file.
    :param **kwargs: (optional) will be passed to torch.onnx.export function.
    '''
    # change to eval mode
    model.eval()
    if not hasattr(model, "ov_infer_engine") or not model.ov_infer_engine.ie_network:
        model.ov_infer_engine = PytorchOpenVINOInference.from_torch(model, input_sample, xml_path)

    model.ov_infer_engine.attach(model)


def exit_openvino(model):
    model.ov_infer_engine.detach(model)
