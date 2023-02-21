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


def fake_tensor_from_spec(tensor_spec: tf.TensorSpec):
    """Fake a `Tensor` from `TensorSpec`."""
    shape = tensor_spec.shape
    dtype = tensor_spec.dtype
    shape = tuple(dim if dim is not None else 1 for dim in shape)
    if shape == () and dtype == tf.bool:
        # This may be the `training` parameter, we should assume it is False
        return False
    return tf.ones(shape=shape, dtype=dtype)
