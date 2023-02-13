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
import numpy as np
import tensorflow as tf
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.common import AcceleratedModel

try:
    import torch
    is_torch_available = True
except Exception as _e:
    # import torch after import tensorflow may cause `OSError` exception on Windows
    is_torch_available = False


class AcceleratedKerasModel(AcceleratedModel, tf.keras.Model):
    """A wrapper class for tf.keras.Model with accelerators."""

    def __init__(self, model, precision=tf.float32):
        super().__init__()
        self.model = model
        self.precision = precision

    def __call__(self, *args, **kwds):
        invalidInputError(
            not kwds.get('training', False),
            "Model of AcceleratedKerasModel is not trainable. Please set `training=False`."
        )
        kwds['training'] = False
        return super().__call__(*args, **kwds)

    def call(self, *inputs):
        output = tf.py_function(self.forward, inputs, Tout=self.precision)
        if hasattr(self, "_output_shape") and self._output_shape is not None:
            output.set_shape(self._output_shape)
        return output

    def forward(self, *inputs):
        inputs = self.on_forward_start(inputs)
        outputs = self.forward_step(*inputs)
        return self.on_forward_end(outputs)

    @staticmethod
    def tensors_to_numpy(tensors, dtype=None):
        if isinstance(dtype, tf.DType):
            dtype = dtype.as_numpy_dtype
        if isinstance(tensors, (list, tuple)):
            return type(tensors)(AcceleratedKerasModel.tensors_to_numpy(tensor, dtype)
                                 for tensor in tensors)
        elif isinstance(tensors, dict):
            return {key: AcceleratedKerasModel.tensors_to_numpy(value, dtype)
                    for key, value in tensors.items()}
        elif isinstance(tensors, tf.Tensor) or is_torch_available and isinstance(tensors,
                                                                                 torch.Tensor):
            if dtype is None:
                return tensors.numpy()
            else:
                return tensors.numpy().astype(dtype)
        elif isinstance(tensors, np.ndarray) and dtype is not None:
            return tensors.astype(dtype)
        else:
            return tensors

    @staticmethod
    def numpy_to_tensors(np_arrays):
        if isinstance(np_arrays, (list, tuple)):
            return type(np_arrays)(AcceleratedKerasModel.numpy_to_tensors(array)
                                   for array in np_arrays)
        elif isinstance(np_arrays, dict):
            return {key: AcceleratedKerasModel.numpy_to_tensors(value)
                    for key, value in np_arrays.items()}
        elif isinstance(np_arrays, np.ndarray):
            return tf.convert_to_tensor(np_arrays)
        else:
            return np_arrays
