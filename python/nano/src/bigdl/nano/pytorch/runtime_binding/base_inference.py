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

from pytorch_lightning import LightningModule
import torch
from functools import partial
import warnings
import math
import numpy as np

BASE_BINDED_COMPONENTS = ['_train_old',
                          '_torch_forward',
                          '_eval_old',
                          'inference']


def on_fit_start(self):
    if "_onnx_on_fit_start" in self.__dict__:
        self._onnx_on_fit_start()
    if "_fx_quantize_on_fit_start" in self.__dict__:
        self._fx_quantize_on_fit_start()
    return self._on_fit_start_old()


def train(self, mode=True):
    if mode:
        if "_onnx_on_train" in self.__dict__:
            self._onnx_on_train(mode)
        if "_fx_quantize_on_train" in self.__dict__:
            self._fx_quantize_on_train(mode)
    return self._train_old(mode)


def eval(self, quantize=None):
    # Note: this order should not be changed
    # 1. run original .eval()
    self._eval_old()
    # 2. recover from onnx
    if "exit_onnx" in self.__dict__:
        self.exit_onnx()
    # 3. apply quantized model if applied
    if "_fx_quantize_eval" in self.__dict__:
        quantize = self._default_inference_quantize if quantize is None else quantize
        self._fx_quantize_eval(quantize)


# inference (new API to unifying users' inference method)
def inference(self,
              input_data,
              batch_size=None,
              sess_options=None,
              backend="onnx",
              quantize=None,
              **kwargs):
    '''
    Inference with/without onnxruntime.
    This method will implicitly build onnxruntime session if it has never been built
    or out-of-date.

    :param input_data: input data for prediction. If backend is set to "onnx",
            the data type should be a numpy ndarray/torch tensor, where the first dim
            should be batch size.
            If backend is NOT set to "onnx", a torch tensor is needed and the pytorch
            forwarding method will be called.
            If there are multiple input, input_data should be a list.
    :param batch_size: int, inferencing batch_size. This value should not affect the
            final inferencing result but will affect resources cost(e.g. memory and time).
            Default to None, which takes all input_data in one batch.
    :param sess_options: ortsess options in ort.SessionOptions type.
    :param backend: str, to set the backend library. "onnx" for onnxruntime, which
            provides lower latency and any other value will make `inference` call
            the pytorch forwarding method.
    :param **kwargs: any other keywords that will be passed to onnx session's building.
    '''

    if isinstance(input_data, list):
        input_sample_list = list(map(lambda x: torch.from_numpy(x), input_data))
    else:
        input_sample_list = [torch.from_numpy(input_data)]

    if backend == "onnx":
        quantize = quantize if quantize is not None else \
            (self.ort_infer_engine.default_eval_precision == "int8")
        if not quantize:
            if not self.ort_infer_engine.ortsess_fp32:
                warnings.warn("Onnxruntime session will be built implicitly,"
                              " this may harm your inference latency.")
                # generate input_sample for ortsess building
                # defaultly set all input to a Tensor(TODO: might be an issue)
                self.eval_onnx(input_sample=tuple(input_sample_list), sess_options=sess_options)
        else:
            if not self.ort_infer_engine.ortsess_int8:
                raise RuntimeError("Please run trainer.quantize again since "
                                   "the quantized onnxruntime session is out-of-date.")
        # generate ort_inputs
        self.eval_onnx(quantize=quantize)
        if batch_size is None:
            # this branch is only to speed up the inferencing when batch_size is set to None.
            return self(*input_sample_list)
        else:
            yhat_list = []
            sample_num = input_sample_list[0].shape[0]  # the first dim should be sample_num
            batch_num = math.ceil(sample_num / batch_size)
            for batch_id in range(batch_num):
                yhat_list.append(self(
                    *tuple(map(lambda x: x[batch_id * batch_size:
                           (batch_id + 1) * batch_size],
                        input_sample_list))))
            # this operation may cause performance degradation
            yhat = np.concatenate(yhat_list, axis=0)
            return yhat
    else:
        # inference w/o onnxruntime (fallback to pytorch native forward)
        quantize = quantize if quantize is not None else self._default_inference_quantize
        self.eval(quantize=quantize)
        with torch.no_grad():
            yhat_list = []
            sample_num = input_sample_list[0].shape[0]  # the first dim should be sample_num
            batch_size = batch_size if batch_size else sample_num
            batch_num = math.ceil(sample_num / batch_size)
            for batch_id in range(batch_num):
                yhat_list.append(self(*map(lambda x: x[batch_id * batch_size:
                                                       (batch_id + 1) * batch_size],
                                           input_sample_list)))
            yhat = torch.cat(yhat_list, axis=0)
            return yhat


def bind_base_inference_rt_methods(pl_model):

    # if all needed method has been binded, return the same model
    if set(BASE_BINDED_COMPONENTS) <= set(dir(pl_model)):
        return pl_model

    if "on_fit_start" in dir(pl_model):
        pl_model._on_fit_start_old = pl_model.on_fit_start
    pl_model._train_old = pl_model.train
    pl_model._torch_forward = pl_model.forward
    pl_model._eval_old = pl_model.eval
    pl_model._default_inference_quantize = False

    pl_model.eval = partial(eval, pl_model)
    pl_model.on_fit_start = partial(on_fit_start, pl_model)
    pl_model.inference = partial(inference, pl_model)
    pl_model.train = partial(train, pl_model)

    return pl_model
