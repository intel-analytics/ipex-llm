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
from bigdl.nano.utils.log4Error import invalidInputError
from ..model import AcceleratedModel


class AcceleratedKerasModel(AcceleratedModel, tf.keras.Model):
    """A wrapper class for tf.keras.Model with accelerators."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, *args, **kwds):
        invalidInputError(
            not kwds.get('training', False),
            "Model of AcceleratedKerasModel is not trainable. Please set `trainning=False`."
        )
        kwds['training'] = False
        return super().__call__(*args, **kwds)

    def call(self, *inputs):
        return tf.py_function(self.forward, inputs, Tout=tf.float32)

    def forward(self, *inputs):
        inputs = self.on_forward_start(inputs)
        outputs = self.forward_step(*inputs)
        return self.on_forward_end(outputs)

    @staticmethod
    def tensors_to_numpy(tensors):
        return tuple(map(lambda x: x.numpy(), tensors))
