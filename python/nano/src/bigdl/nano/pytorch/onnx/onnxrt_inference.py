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
from functools import partial
import warnings
import torch
import math
import numpy as np


ONNXRT_BINDED_COMPONENTS = ['_ortsess_up_to_date',
                            '_ortsess',
                            '_build_ortsess',
                            'update_ortsess',
                            'on_fit_start',
                            'inference']


# internal function to build an ortsess
def _build_ortsess(self,
                   input_sample=None,
                   file_path="model.onnx",
                   sess_options=None,
                   **kwargs):
    '''
    Internal function to build a ortsess and bind to the lightningmodule.

    :param input_sample: torch.Tensor for the model tracing.
    :param file_path: The path to save onnx model file.
    :param sess_options: ortsess options in ort.SessionOptions type
    :param **kwargs: will be passed to torch.onnx.export function.
    '''

    if input_sample is None and self.example_input_array is not None:
        input_sample = self.example_input_array  # use internal example_input_array
    else:
        self.example_input_array = input_sample  # set example_input_array for future usage

    assert input_sample is not None,\
        'You should set either input_sample or self.example_input_array'

    default_onnx_export_args = {'export_params': True,
                                'opset_version': 10,
                                'do_constant_folding': True,
                                'input_names': ['input'],
                                'output_names': ['output'],
                                'dynamic_axes': {'input': {0: 'batch_size'},
                                                 'output': {0: 'batch_size'}}}
    default_onnx_export_args.update(kwargs)

    self.to_onnx(file_path,
                 input_sample,
                 **default_onnx_export_args)

    self._ortsess = ort.InferenceSession(file_path, sess_options=sess_options)
    self._ortsess_up_to_date = True


# external method to update(& rebuild) ortsess
def update_ortsess(self,
                   input_sample=None,
                   file_path="model.onnx",
                   sess_options=None,
                   **kwargs):
    '''
    Update the onnxruntime session options and rebuild the session.
    Users may also want to call this method before `inference(..., onnx=True`)`
    to avoid implicit building.

    :param input_sample: torch.Tensor for the model tracing.
    :param file_path: The path to save onnx model file.
    :param sess_options: ortsess options in ort.SessionOptions type.
    :param **kwargs: will be passed to torch.onnx.export function.
    '''
    self._build_ortsess(input_sample=input_sample,
                        file_path=file_path,
                        sess_options=sess_options,
                        **kwargs)


# inference (new API to unifying users' inference method)
def inference(self,
              input_data,
              batch_size=None,
              sess_options=None,
              backend="onnx",
              **kwargs):
    '''
    Inference with/without onnxruntime.
    This method will implicitly build onnxruntime session if it has never been built
    or out-of-date.

    :param input_data: input data for prediction. If backend is set to "onnx",
            the data type should be a numpy ndarray, where the first dim should be batch size.
            If backend is NOT set to "onnx", a torch tensor is needed and the pytorch
            forwarding method will be called.
    :param batch_size: int, inferencing batch_size. This value should not affect the
            final inferencing result but will affect resources cost(e.g. memory and time).
            Default to None, which takes all input_data in one batch.
    :param sess_options: ortsess options in ort.SessionOptions type.
    :param backend: str, to set the backend library. "onnx" for onnxruntime, which
            provides lower latency and any other value will make `inference` call
            the pytorch forwarding method.
    :param **kwargs: any other keywords that will be passed to onnx session's building.
    '''

    if backend == "onnx":
        if not self._ortsess_up_to_date:
            warnings.warn("Onnxruntime session will be built implicitly,"
                          " this may harm your inference latency.")
            input_sample = torch.Tensor(input_data)
            self._build_ortsess(input_sample=input_sample,
                                file_path="model.onnx",
                                sess_options=sess_options,
                                **kwargs)
        input_name = self._ortsess.get_inputs()[0].name
        if batch_size is None:
            # this branch is only to speed up the inferencing when batch_size is set to None.
            ort_inputs = {input_name: input_data}
            ort_outs = self._ortsess.run(None, ort_inputs)
            return ort_outs[0]
        else:
            yhat_list = []
            sample_num = input_data.shape[0]  # the first dim should be sample_num
            batch_num = math.ceil(sample_num / batch_size)
            for batch_id in range(batch_num):
                ort_inputs = {input_name: input_data[batch_id * batch_size:
                                                     (batch_id + 1) * batch_size]}
                ort_outs = self._ortsess.run(None, ort_inputs)
                yhat_list.append(ort_outs[0])
            # this operation may cause performance degradation
            yhat = np.concatenate(yhat_list, axis=0)
            return yhat
    else:
        # inference w/o onnxruntime (fallback to pytorch native forward)
        self.eval()
        with torch.no_grad():
            yhat_list = []
            sample_num = input_data.shape[0]  # the first dim should be sample_num
            batch_size = batch_size if batch_size else sample_num
            batch_num = math.ceil(sample_num / batch_size)
            for batch_id in range(batch_num):
                yhat_list.append(self(input_data[batch_id * batch_size:
                                                 (batch_id + 1) * batch_size]))
            yhat = torch.cat(yhat_list, axis=0)
            return yhat


# on_fit_start (LightningModule method overwrite)
def on_fit_start(self):
    self._ortsess_up_to_date = False


def bind_onnxrt_methods(pl_model: LightningModule):
    # class type check
    assert isinstance(pl_model, LightningModule),\
        "onnxruntime support is only valid for a LightningModule."

    # check conflicts
    for component in ONNXRT_BINDED_COMPONENTS:
        if component in dir(pl_model):
            warnings.warn(f"{component} method/property will be replaced.")

    # additional attributes
    pl_model._ortsess_up_to_date = False  # indicate if we need to build ortsess again
    pl_model._ortsess = None  # ortsess instance

    # additional methods
    pl_model._build_ortsess = partial(_build_ortsess, pl_model)
    pl_model.update_ortsess = partial(update_ortsess, pl_model)
    pl_model.on_fit_start = partial(on_fit_start, pl_model)
    pl_model.inference = partial(inference, pl_model)

    return pl_model
