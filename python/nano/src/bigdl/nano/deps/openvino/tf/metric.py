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


import tensorflow as tf
from tensorflow.keras.metrics import Metric
from bigdl.nano.utils.common import invalidInputError

from ..core.metric import BaseOpenVINOMetric


class KerasOpenVINOMetric(BaseOpenVINOMetric):
    def __init__(self, metric, higher_better=True):
        invalidInputError(isinstance(metric, Metric),
                          "Please provide an instance of tf.keras.metrics.Metric")
        super().__init__(metric, higher_better)

    def stack(self, output):
        return tf.stack(output)

    def update(self, output, target):
        def numpy_to_tensors(np_arrays):
            return tuple(map(lambda x: tf.convert_to_tensor(x), np_arrays))
        output = numpy_to_tensors(output)
        target = numpy_to_tensors(target)
        return super().update(output, target)

    def reset(self):
        self._scores = []

    def to_scalar(self, score):
        return float(score)
