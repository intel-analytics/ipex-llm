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
import os
from ..core.onnxruntime_model import ONNXRuntimeModel
from bigdl.nano.utils.inference.pytorch.model import AcceleratedLightningModule
from bigdl.nano.utils.inference.pytorch.model_utils import export_to_onnx, get_forward_args


class PytorchONNXRuntimeModel(ONNXRuntimeModel, AcceleratedLightningModule):
    '''
        This is the accelerated model for pytorch and onnxruntime.
        All the external API is based on Trainer, so what we have here is
        basically internal APIs and subject to change.

        This PytorchONNXRuntimeModel will serve for all precision models.
    '''

    def __init__(self, model, input_sample=None):
        """
        Create a ONNX Runtime model from pytorch.

        :param model: 1. Pytorch model to be converted to ONNXRuntime for inference
                      2. Path to ONNXRuntime saved model.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached,
                             defaults to None.
        """
        # Typically, when model is int8, we use this path
        # TODO: self._forward_args should be set externally
        onnx_path = model
        if isinstance(model, torch.nn.Module):
            # Typically, when model is fp32, we use this path
            # TODO: expose ONNX export parameters to users
            export_to_onnx(model, input_sample=input_sample, onnx_path='tmp.onnx')
            onnx_path = 'tmp.onnx'
        AcceleratedLightningModule.__init__(self, None)
        ONNXRuntimeModel.__init__(self, onnx_path)
        if os.path.exists('tmp.onnx'):
            os.remove('tmp.onnx')

    def on_forward_start(self, inputs):
        if self.ortsess is None:
            raise RuntimeError(
                "Please create an instance by PytorchONNXRuntimeModel()"
            )
        inputs = self.tensors_to_numpy(inputs)
        return inputs

    def on_forward_end(self, outputs):
        outputs = self.numpy_to_tensors(outputs)
        return outputs

    @staticmethod
    def load(path):
        """
        Load an OpenVINO model for inference.

        :param path: Path to model to be loaded.
        :return: PytorchOpenVINOModel model for OpenVINO inference.
        """
        assert path.split('.')[-1] == "onnx", "Path of onnx model must be with '.onnx' suffix."
        return PytorchONNXRuntimeModel(path)
