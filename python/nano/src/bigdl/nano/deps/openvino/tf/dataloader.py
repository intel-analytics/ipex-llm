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


from openvino.tools.pot import DataLoader
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import BatchDataset

from bigdl.nano.utils.log4Error import invalidInputError


class KerasOpenVINODataLoader(DataLoader):
    def __init__(self, dataset, collate_fn=None):
        invalidInputError(isinstance(dataset, tf.data.Dataset),
                          "Please provide an instance of tf.data.Dataset")
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.next_index = 0
        self.iter = iter(self.dataset)

    def __getitem__(self, index):
        if index > len(self):
            invalidInputError(False, f"index out of bounds, index:{index}, length:{len(self)}")

        if index < self.next_index:
            self.next_index = 0
            self.iter = iter(self.dataset)
        if index > self.next_index:
            for _ in range(index - self.next_index):
                _ = next(self.iter)
            self.next_index = index

        data = next(self.iter)
        if self.collate_fn:
            data = self.collate_fn(data)
        self.next_index += 1
        return data

    def __len__(self):
        return len(self.dataset)
