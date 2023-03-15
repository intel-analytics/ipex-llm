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

from bigdl.nano.utils.common import invalidInputError


class KerasOpenVINODataLoader(DataLoader):
    def __init__(self, x, y, collate_fn=None):
        self.x = x
        self.y = y
        self.collate_fn = collate_fn
        self._reset()

    def __getitem__(self, index):
        if index > len(self):
            invalidInputError(False, f"index out of bounds, index:{index}, length:{len(self)}")

        if index < self.next_index:
            self._reset()
        if index > self.next_index:
            for _ in range(index - self.next_index):
                self._next()

        data = self._next()
        if self.collate_fn:
            data = self.collate_fn(data)
        return data

    def __len__(self):
        return len(self.x)

    def _reset(self):
        self.next_index = 0
        if isinstance(self.x, tf.data.Dataset):
            self.x_iter = iter(self.x)
        else:
            self.x_iter = iter(self.x)
            self.y_iter = iter(self.y)

    def _next(self):
        self.next_index += 1
        if isinstance(self.x, tf.data.Dataset):
            return next(self.x_iter)
        else:
            return next(self.x_iter), next(self.y_iter)
