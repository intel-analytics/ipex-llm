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


from typing import Union, Optional, Sequence, Dict

import numpy as np
import tensorflow as tf

from bigdl.nano.utils.common import invalidInputError


DATA_TYPE = Union[tf.Tensor, np.ndarray]
DTYPE = Optional[Union[tf.DType, np.dtype]]


def convert_all(inputs: Union[Dict[str, DATA_TYPE], Sequence[DATA_TYPE]],
                types: Union[Dict[str, str], Sequence[str], str],
                dtypes: Union[Dict[str, DTYPE], Sequence[DTYPE], DTYPE] = None,
                ):
    """
    Convert all input tf.Tensor/np.ndarray to specified format.

    Usage:
    ```
    x = np.random.random((10, 10))
    y = np.random.random((10, 10))
    x, y = convert_all([x, y], types="tf", dtypes=tf.float32)

    x = np.random.random((10, 10))
    y = np.random.random((10, 10))
    x, y = convert_all({"x":x, "y": y}, types="tf", dtypes={"x": tf.float32, "y": tf.bfloat16})

    x = np.random.random((10, 10))
    y = tf.random.normal((10, 10))
    x, y = convert_all([x, y], types=["tf", "numpy"], dtypes=[tf.float32, np.float32])
    ```

    :param input_: A list/dict of tf.Tensor/np.ndarray to convert.
    :param type_: (A list/dict of) target type, "tf" means tf.Tensor, "numpy" means np.ndarray.
    :param dtype_: (A list/dict of) target dtype.
    :return The convert result.
    """
    if isinstance(inputs, Sequence):
        return _convert_list(inputs, types, dtypes)     # type: ignore
    else:
        return _convert_dict(inputs, types, dtypes)     # type: ignore


def _convert_list(inputs: Sequence[DATA_TYPE],
                  types: Union[Sequence[str], str],
                  dtypes: Union[Sequence[DTYPE], DTYPE] = None):
    result = []
    for idx, input_ in enumerate(inputs):
        type_ = types[idx] if isinstance(types, (list, tuple)) else types
        dtype_ = dtypes[idx] if isinstance(dtypes, (list, tuple)) else dtypes
        result.append(convert(input_, type_, dtype_))
    return result


def _convert_dict(inputs: Dict[str, DATA_TYPE],
                  types: Union[Dict[str, str], str],
                  dtypes: Union[Dict[str, DTYPE], DTYPE] = None):
    result = {}
    for name, input_ in inputs.items():
        type_ = types[name] if isinstance(types, dict) else types
        dtype_ = dtypes[name] if isinstance(dtypes, dict) else dtypes
        result[name] = convert(input_, type_, dtype_)
    return result


def convert(input_: DATA_TYPE,
            type_: str,
            dtype_: DTYPE = None):
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
    :return The results of conversion.
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


# deprecated
def tensors_to_numpy(tensors, dtype=None):
    """Convert tf Tensor(s) to numpy ndarray(s)."""
    if isinstance(dtype, tf.DType):
        dtype = dtype.as_numpy_dtype
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_numpy(tensor, dtype) for tensor in tensors)
    elif isinstance(tensors, dict):
        return {key: tensors_to_numpy(value, dtype)
                for key, value in tensors.items()}
    elif isinstance(tensors, tf.Tensor):
        if dtype is None:
            return tensors.numpy()
        else:
            return tensors.numpy().astype(dtype)
    elif isinstance(tensors, np.ndarray) and dtype is not None:
        return tensors.astype(dtype)
    else:
        return tensors


# deprecated
def numpy_to_tensors(np_arrays):
    """Convert numpy ndarray(s) to tf Tensor(s)."""
    if isinstance(np_arrays, (list, tuple)):
        return type(np_arrays)(numpy_to_tensors(array) for array in np_arrays)
    elif isinstance(np_arrays, dict):
        return {key: numpy_to_tensors(value)
                for key, value in np_arrays.items()}
    elif isinstance(np_arrays, np.ndarray):
        return tf.convert_to_tensor(np_arrays)
    else:
        return np_arrays


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
#                 yield tensors_to_numpy(batch, self.dtype)
#         else:
#             for x, y in zip(self.x, self.y):
#                 x, y = tensors_to_numpy((x, y), self.dtype)
#                 yield np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
