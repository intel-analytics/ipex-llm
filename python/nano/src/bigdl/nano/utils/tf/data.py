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


from typing import List, Union, Optional, Sequence

import numpy as np
import tensorflow as tf

from bigdl.nano.utils.common import invalidInputError


DTYPE = Union[tf.DType, np.dtype]


def convert_all(inputs: Sequence[Union[tf.Tensor, np.ndarray]],
                types=Union[List[Optional[str]], Optional[str]],
                dtypes=Union[List[Optional[DTYPE]], Optional[DTYPE]],
                ):
    """
    Convert all input tf.Tensor/np.ndarray to specified format.

    Usage:
    ```
    x = np.random.random((10, 10))
    y = np.random.random((10, 10))
    x, y = convert_all((x, y), types="tf", dtypes=tf.float32)

    x = np.random.random((10, 10))
    y = tf.random.normal((10, 10))
    x, y = convert_all((x, y), types=["tf", "numpy"], dtypes=[tf.float32, np.float32])
    ```

    :param input_: The tf.Tensor/np.ndarray to convert.
    :param type_: (A list of) target type, "tf" means tf.Tensor, "numpy" means np.ndarray.
    :param dtype_: (A list of) target dtype.
    :return The convert result.
    """
    if not isinstance(types, list):
        types = [types] * len(inputs)
    if not isinstance(dtypes, list):
        dtypes = [dtypes] * len(inputs)

    return [convert(input_, type_, dtype_)
            for input_, type_, dtype_ in zip(inputs, types, dtypes)]


def convert(input_: Union[tf.Tensor, np.ndarray],
            type_: Optional[str],
            dtype_: Optional[DTYPE]):
    """
    Convert tf.Tensor/np.ndarray to specified format.

    Usage:
    ```
    x = np.random.random((10, 10))
    convert(x, type_="tf", dtype_=tf.float32)

    x = tf.random.normal((10, 10))
    convert(x, type_="numpy", dtype_=np.float32)
    ```

    :param input_: The tf.Tensor/np.ndarray to convert.
    :param type_: The target type, "tf" means tf.Tensor, "numpy" means np.ndarray.
    :param dtype_: The target dtype.
    :return The convert result.
    """
    # todo: we should also convert `int`, `float`, `bool` to scalar Tensor/numpy
    if type_ == "tf":
        if isinstance(input_, np.ndarray):
            return tf.convert_to_tensor(input_, dtype=dtype_)
        elif isinstance(input_, tf.Tensor):
            if dtype_ is not None and input_.dtype != dtype_:
                return tf.cast(input_, dtype=dtype_)
            else:
                return input_
        else:
            invalidInputError(False, f"Unkonwn type: {type(input_)}")
    elif type_ == "numpy":
        if isinstance(input_, tf.Tensor):
            input_ = input_.numpy()
        if isinstance(input_, np.ndarray):
            if dtype_ is not None and input_.dtype != dtype_:
                return input_.astype(dtype_)
            else:
                return input_
        else:
            invalidInputError(False, f"Unkonwn type: {type(input_)}")
    else:
        invalidInputError(False, f"Invalid target type: {type_}")


# todo: Add a common keras dataset
# class KerasDataset():
#     def __init__(self,
#                  x,
#                  y,
#                  collate_fn=None,
#                  dtype=tf.float32) -> None:
#         self.x = x,
#         self.y = y,
#         self.collate_fn = collate_fn
#         self.dtype=dtype

#     def __getitem__(self, index):
#         if index > len(self):
#             invalidInputError(False, f"index out of bounds, index:{index}, length:{len(self)}")

#         if index < self.next_index:
#             self._reset()
#         if index > self.next_index:
#             for _ in range(index - self.next_index):
#                 self._next()

#         data = self._next()
#         if self.collate_fn:
#             data = self.collate_fn(data)
#         return data

#     def __len__(self):
#         return len(self.x)

#     def _reset(self):
#         self.next_index = 0
#         if isinstance(self.x, tf.data.Dataset):
#             self.x_iter = iter(self.x)
#         else:
#             self.x_iter = iter(self.x)
#             self.y_iter = iter(self.y)

#     def _next(self):
#         self.next_index += 1
#         if isinstance(self.x, tf.data.Dataset):
#             return next(self.x_iter)
#         else:
#             return next(self.x_iter), next(self.y_iter)

#     def __iter__(self):
#         if isinstance(self.x, tf.data.Dataset):
#             for batch in self.x.batch(1):
#                 yield AcceleratedKerasModel.tensors_to_numpy(batch, self.dtype)
#         else:
#             for x, y in zip(self.x, self.y):
#                 x, y = AcceleratedKerasModel.tensors_to_numpy((x, y), self.dtype)
#                 yield np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
