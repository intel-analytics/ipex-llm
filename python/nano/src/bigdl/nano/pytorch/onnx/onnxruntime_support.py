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
import onnxruntime as ort
from functools import wraps
import torch
import warnings
import math
import numpy as np

def onnxruntime(override_predict_step=True):
    '''
    `onnxruntime` is a decorator that extend a LightningModule to support onnxruntime inference
    in an easy way.

    With this decorator used, a LightningModule will be revised as following:
    - a new method `inference(input_data, batch_size=None, backend="onnx", **kwargs)` will
      be added for onnx inference.
    - `predict_step` will be override if override_predict_step is set to True.

    :param override_predict_step: bool, decide if `predict_step` will be override.
    '''

    def onnxruntime_decorator(cls):
        # class type check
        assert issubclass(cls, LightningModule),\
            "onnxruntime decorator is only valid for a LightningModule."

        # additional attributes
        cls._ortsess_up_to_date = False  # indicate if we need to build ortsess again
        cls._ortsess = None  # ortsess instance

        # _build_ortsess
        def _build_ortsess(self, input_sample=None, filepath="model.onnx", **kwargs):
            if input_sample is None and self.example_input_array is not None:
                input_sample = self.example_input_array
            self.to_onnx(filepath, input_sample, export_params=True, **kwargs)
            self._ortsess = ort.InferenceSession(filepath)
            self._ortsess_up_to_date = True
        cls._build_ortsess = _build_ortsess

        # inference
        def inference(self, input_data, batch_size=None, backend="onnx", **kwargs):
            '''
            :param input_data: input data for prediction. If backend is set to "onnx", 
                   the data type should be a numpy ndarray.  If backend is NOT set to "onnx",
                   a torch tensor is needed and the pytorch forwarding method will be called.
            :param backend: str, to set the backend library. "onnx" for onnxruntime, which
                   provides lower latency and any other value will make `inference` call
                   the pytorch forwarding method.
            :param batch_size: int, inferencing batch_size. This value should not affect the
                   final inferencing result but will affect resources cost(e.g. memory and time).
                   Default to None, which takes all input_data in one batch.
            :param **kwargs: any other keywords that will be passed to onnx session's building.
            '''
            if backend == "onnx":
                if not self._ortsess_up_to_date:
                    warnings.warn("Onnxruntime session is built lazily,"
                                  " this may harm your inference latency.")
                    input_sample = torch.Tensor(input_data)
                    self._build_ortsess(input_sample=input_sample, **kwargs)
                input_name = self._ortsess.get_inputs()[0].name
                if batch_size is None:  # this branch is only to speed up the inferencing
                    ort_inputs = {input_name: input_data}
                    ort_outs = self._ortsess.run(None, ort_inputs)
                    return ort_outs[0]
                else:
                    yhat_list = []
                    sample_num = input_data.shape[0]  # the first dim should be sample_num
                    batch_num = math.ceil(sample_num/batch_size)
                    for batch_id in range(batch_num):
                        ort_inputs = {input_name: input_data[batch_id*batch_size:\
                            (batch_id+1)*batch_size]}
                        ort_outs = self._ortsess.run(None, ort_inputs)
                        yhat_list.append(ort_outs[0])
                    yhat = np.concatenate(yhat_list, axis=0)
                    return yhat
            else:
                self.eval()
                with torch.no_grad():
                    yhat_list = []
                    sample_num = input_data.shape[0]  # the first dim should be sample_num
                    batch_size = batch_size if batch_size else sample_num
                    batch_num = math.ceil(sample_num/batch_size)
                    for batch_id in range(batch_num):
                        yhat_list.append(self(input_data[batch_id*batch_size:\
                            (batch_id+1)*batch_size]))
                    yhat = np.concatenate(yhat_list, axis=0)
                    return self(input_data)
        cls.inference = inference

        # on_fit_start
        def on_fit_start_additional(function):
            def wrapped(*args, **kwargs):
                args[0]._ortsess_up_to_date = False
                return function(*args, **kwargs)
            return wrapped
        cls.on_fit_start = on_fit_start_additional(cls.on_fit_start)

        # predict_step
        if override_predict_step:
            def predict_step(self, batch, batch_idx):
                return self.inference(batch[0].numpy())
            cls.predict_step = predict_step

        return cls

    return onnxruntime_decorator
