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
from openvino.tools.pot import Metric
import numpy as np


class BaseOpenVINOMetric(Metric):
    def __init__(self, metric, higher_better=True):
        super().__init__()
        self.metric = metric
        self._higher_better = higher_better
        self._pred_list = []
        self._target_list = []
        self._name = type(self.metric).__name__

    @property
    def value(self):
        score = self.metric(self._pred_list[-1], self._target_list[-1])
        return {self._name: [self.to_scalar(score)]}

    @property
    def avg_value(self):
        preds = self.stack(self._pred_list)
        targets = self.stack(self._target_list)
        score = self.metric(preds, targets)
        return {self._name: self.to_scalar(score)}

    def stack(self, output):
        return np.stack(output)

    def update(self, output, target):
        self._pred_list.extend(output)
        self._target_list.extend(target)

    def reset(self):
        self._scores = []

    @property
    def higher_better(self):
        return self._higher_better

    def to_scalar(self, score):
        return score.item()

    def get_attributes(self):
        name = type(self.metric).__name__
        direction = 'higher-better' if self.higher_better else 'higher-worse'
        return {self._name: {"type": self._name, "direction": direction}}
