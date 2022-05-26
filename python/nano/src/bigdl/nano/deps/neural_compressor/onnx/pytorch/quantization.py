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
from bigdl.nano.utils.log4Error import invalidInputError
from ..quantization import BaseONNXRuntimeQuantization
from ...pytorch.quantization import PytorchQuantization
from .metric import PytorchONNXRuntimeINCMetic
import torch


class PytorchONNXRuntimeQuantization(BaseONNXRuntimeQuantization, PytorchQuantization):
    def __init__(self, framework='onnxrt_qlinear', **kwargs):
        """
        Create a Intel Neural Compressor Quantization object for ONNXRuntime in Pytorch.
        """
        kwargs['framework'] = framework
        super().__init__(**kwargs)
        self._inc_metric_cls = PytorchONNXRuntimeINCMetic

    def _pre_execution(self, model, calib_dataloader=None, metric=None):

        if calib_dataloader:

            def func(data):
                # TODO: only x, y are supported here for onnx quantization
                x, y = zip(*data)
                if isinstance(x[0], torch.Tensor):
                    x = torch.stack(x, dim=0).numpy()
                if isinstance(y[0], torch.Tensor):
                    y = torch.stack(y, dim=0).numpy()
                return x, y

            # add a collate_fn to transform torch dataloader to a numpy dataloader
            calib_dataloader = deepcopy(calib_dataloader)
            calib_dataloader.collate_fn = func

        if isinstance(model, PytorchONNXRuntimeModel):
            model = model.onnx_model

        return model, calib_dataloader, metric

    def _post_execution(self, q_model):
        # TODO Don't save, directly use q_model to create runtime
        with TemporaryDirectory() as dir:
            saved_onnx = Path(dir) / 'tmp.onnx'
            q_model.save(saved_onnx)
            return PytorchONNXRuntimeModel(str(saved_onnx))
