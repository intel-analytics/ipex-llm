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
import numpy as np
from functools import wraps


def nano_bf16(func):
    """A decorator to realize mixed precision on customized training loop."""
    # todo check the func signature
    @wraps(func)
    def wrapper(*inner_args):
        new_args = []
        for arg in inner_args:
            if isinstance(arg, (tf.Tensor, np.ndarray)):
                arg = tf.cast(arg, tf.bfloat16)
            new_args.append(arg)
        return func(*new_args)
    return wrapper
