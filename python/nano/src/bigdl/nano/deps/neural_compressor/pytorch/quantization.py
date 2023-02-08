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
from typing import Callable
from collections import Iterable
from bigdl.nano.utils.common import invalidInputError
from ..core import BaseQuantization
from .utils import _check_loader
from .metric import PytorchINCMetric
from .quantized_model import PytorchQuantizedModel
from torchmetrics import Metric
import torch
from bigdl.nano.pytorch.utils import patch_attrs_from_model_to_object


class PytorchQuantization(BaseQuantization):
    def __init__(self, framework='pytorch_fx', thread_num=None, **kwargs):
        """
        Create a Intel Neural Compressor Quantization object for Pytorch.
        """
        kwargs['framework'] = framework
        super().__init__(**kwargs)
        self._inc_metric_cls = PytorchINCMetric
        self.thread_num = thread_num

    def _pre_execution(self, model, calib_dataloader=None, metric=None):
        if isinstance(model, torch.nn.Module):
            self.original_model = model
        return model, calib_dataloader, metric

    def _post_execution(self, q_model):
        quantized_model = PytorchQuantizedModel(q_model, self.thread_num)
        if hasattr(self, "original_model"):
            # patch original model's attr to current new model
            patch_attrs_from_model_to_object(self.original_model, quantized_model)
            del self.original_model
        return quantized_model

    @property
    def valid_frameworks(self):
        return ('pytorch_fx', 'pytorch', 'pytorch_ipex')

    def sanity_check_before_execution(self, model, calib_dataloader, metric):
        invalidInputError(isinstance(model, torch.nn.Module),
                          "model should be an instance of torch.nn.Module.")
        if calib_dataloader:
            _check_loader(model=model, loader=calib_dataloader, metric=metric)
            invalidInputError(isinstance(calib_dataloader, Iterable),
                              "Only iterable class is supported.")
        if metric:
            invalidInputError(
                isinstance(metric, Metric) or isinstance(metric, Callable),
                errMsg="Metric of type {} is invalid".format(type(metric)),
                fixMsg="Use instance of `torchmetrics.Metric` instead."
            )
        super().sanity_check_before_execution(model, calib_dataloader, metric)
