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


def rollback_tf_dataset(tf_data):
    x, y= tf_data.map(lambda x, y: x), tf_data.map(lambda x, y: y)
    x = np.concatenate(tuple(x.as_numpy_iterator()), axis=0)
    y = np.concatenate(tuple(y.as_numpy_iterator()), axis=0)
    return x, y


def np_to_data_creator(tf_data):
    def data_creator(config, batch_size):
        data_len = tf_data[0].shape[0]
        data = tf.data.Dataset.from_tensor_slices((tf_data))
        data = data.cache().shuffle(data_len).batch(batch_size)
        return data.prefetch(tf.data.AUTOTUNE)
    return data_creator
