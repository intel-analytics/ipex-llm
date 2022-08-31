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
from ..core import BaseINCMetric
import tensorflow as tf


class TensorflowINCMetric(BaseINCMetric):
    def stack(self, preds, labels):

        # calculate accuracy
        preds = tf.stack(preds)
        labels = tf.stack(labels)
        return preds, labels

    def result(self):
        # calculate accuracy
        preds, labels = self.stack(self.pred_list, self.label_list)
        accuracy = self.metric(labels, preds)
        return self.to_scalar(accuracy)

    def to_scalar(self, tensor):
        return tensor.numpy()
