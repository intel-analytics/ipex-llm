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
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np


class NanoMetric(object):
    def __init__(self, metric: Callable):
        self.metric = metric

    def __call__(self, model: nn.Module, data_loader: DataLoader):
        with torch.no_grad():
            metric_list = []
            sample_num = 0
            for data_input, target in data_loader:
                metric_value = self.metric(model(data_input), target)
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.numpy()
                metric_list.append(
                    metric_value * data_input.shape[0])
                sample_num += data_input.shape[0]
            return np.sum(metric_list) / sample_num
