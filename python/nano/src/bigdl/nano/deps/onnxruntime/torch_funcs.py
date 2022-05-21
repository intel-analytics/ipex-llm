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
from .base_onnxruntime import BaseORTInference, save_onnx_to_file
from bigdl.nano.utils.inference.pytorch_base_inference import PytorchBaseInference, get_forward_args
from bigdl.nano.utils.inference.pytorch_base_inference import export as export_to_onnx
from bigdl.nano.utils.log4Error import invalidInputError
import torch


class PytorchORTInference(BaseORTInference, PytorchBaseInference):
    def __init__(self):
        BaseORTInference.__init__(self)
        PytorchBaseInference.__init__(self)
        self.default_eval_precision = "fp32"

    def build_ortsess_fp32(self, model, input_sample=None, onnx_path="model.onnx",
                           sess_options=None):
        export_to_onnx(model, input_sample=input_sample,
                       onnx_path=onnx_path, dynamic_axes=True)
        self._forward_args = get_forward_args(model)
        self.onnx_filepath_fp32 = onnx_path
        self._build_ortsess_fp32(sess_options=sess_options)

    def build_ortsess_int8(self, model, onnx_model, onnx_path="model_int8.onnx",
                           sess_options=None):
        save_onnx_to_file(onnx_model, onnx_path)
        self._forward_args = get_forward_args(model)
        self.onnx_filepath_int8 = onnx_path
        self._build_ortsess_int8(sess_options=sess_options)

    def attach(self, model, precision="fp32"):
        # set original `train` and `forward`
        if not getattr(model, "_torch_train", None):
            model._torch_train = model.train
            model._torch_forward = model.forward

        # bind new forward and forward_int8
        if precision == "fp32":
            model.forward = partial(self.forward, model)
        if precision == "int8":
            model.forward = partial(self.forward_int8, model)

        # bind new `train`
        model.train = partial(self.train, model)

    def detach(self, model):
        model.forward = model._torch_forward

    def on_train_start(self, model, mode):
        self.detach(model)
        if mode:
            self.reset(model)


def eval_onnx(model, input_sample=None, file_path="model.onnx",
              sess_options=None, quantize=None):
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
    model.eval(quantize=False)

    # check default evaluation precision
    if quantize is None:
        quantize = (model.ort_infer_engine.default_eval_precision == "int8")

    if quantize:
        first_time_after_fitting = not model.ort_infer_engine.ortsess_int8
        if first_time_after_fitting:
            # not in8 ortsess, can't help, return an error
            invalidInputError(False,
                              "Please call trainer.quantize before using eval_onnx(quantize=True)")
        model.ort_infer_engine.attach(model, precision="int8")
    else:
        first_time_after_fitting = not model.ort_infer_engine.ortsess_fp32
        if first_time_after_fitting:
            # no fp32 ortsess, then build one
            model.ort_infer_engine.build_ortsess_fp32(model=model,
                                                      input_sample=input_sample,
                                                      onnx_path=file_path,
                                                      sess_options=sess_options)
        model.ort_infer_engine.attach(model, precision="fp32")


def exit_onnx(model):
    model.ort_infer_engine.detach(model)


def to_quantized_onnx(model, file_path):
    if model.ort_infer_engine.onnx_model_int8:
        save_onnx_to_file(model.ort_infer_engine.onnx_model_int8, file_path)
    else:
        invalidInputError(False,
                          "Please run trainer.quantize again since "
                          "the quantized onnxruntime session is out-of-date.")
