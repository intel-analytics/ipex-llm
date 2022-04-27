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
from pytorch_lightning import LightningModule


def bind_onnxrt_methods(pl_model: LightningModule, q_onnx_model=None, sess_options=None):
    from . import torch_funcs
    # add an ort_infer_engine to control the runtime binding
    if not hasattr(pl_model, "ort_infer_engine"):
        pl_model.ort_infer_engine = torch_funcs.PytorchORTInference()
    if q_onnx_model:
        pl_model.ort_infer_engine.build_ortsess_int8(pl_model, q_onnx_model.model)
        pl_model.ort_infer_engine.default_eval_precision = "int8"
    pl_model.eval_onnx = partial(torch_funcs.eval_onnx, pl_model)
    pl_model.exit_onnx = partial(torch_funcs.exit_onnx, pl_model)
    pl_model.to_quantized_onnx = partial(torch_funcs.to_quantized_onnx, pl_model)
    return pl_model


def PytorchONNXRuntimeModel(model, input_sample=None):
    """
        Create a ONNX Runtime model from pytorch.

        :param model: 1. Pytorch model to be converted to ONNXRuntime for inference
                      2. Path to ONNXRuntime saved model.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached,
                             defaults to None.

        :return: A PytorchONNXRuntimeModel instance
        """
    from .pytorch.pytorch_onnxruntime_model import PytorchONNXRuntimeModel
    return PytorchONNXRuntimeModel(model, input_sample)


def load_onnxruntime_model(path):
    from .pytorch.pytorch_onnxruntime_model import PytorchONNXRuntimeModel
    return PytorchONNXRuntimeModel._load(path)
