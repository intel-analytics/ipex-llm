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
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from bigdl.nano.deps.onnxruntime.pytorch.pytorch_onnxruntime_model import PytorchONNXRuntimeModel
from ..quantization import BaseONNXRuntimeQuantization
from ...pytorch import PytorchQuantization
from .metric import PytorchONNXRuntimeINCMetic
import torch


class PytorchONNXRuntimeQuantization(BaseONNXRuntimeQuantization, PytorchQuantization):
    def __init__(self, framework='onnxrt_qlinear', **kwargs):
        """
        Create a Intel Neural Compressor Quantization object for ONNXRuntime in Pytorch.
        """
        kwargs['framework'] = framework
        self.session_options = kwargs.pop('onnxruntime_session_options', None)
        super().__init__(**kwargs)
        self._inc_metric_cls = PytorchONNXRuntimeINCMetic

    def _pre_execution(self, model, calib_dataloader=None, metric=None):

        if calib_dataloader:

            def numpy_collate_fn_wrapper(func):
                def transform_tensor_to_numpy(item):
                    if isinstance(item, torch.Tensor):
                        return item.numpy()
                    return item

                def collate_fn(batch):
                    res = func(batch)
                    return tuple(map(lambda x: transform_tensor_to_numpy(x), res))
                return collate_fn

            # add a collate_fn to transform torch dataloader to a numpy dataloader
            calib_dataloader = deepcopy(calib_dataloader)
            calib_dataloader.collate_fn = numpy_collate_fn_wrapper(calib_dataloader.collate_fn)

        if isinstance(model, PytorchONNXRuntimeModel):
            model = model.onnx_model

        return model, calib_dataloader, metric

    def _post_execution(self, q_model):
        return PytorchONNXRuntimeModel(q_model.model,
                                       onnxruntime_session_options=self.session_options)
