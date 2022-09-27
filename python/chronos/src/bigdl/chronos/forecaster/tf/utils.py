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


__all__ = ['np_to_data_creator',
           'tsdata_to_data_creator',
           'np_to_tfdataset',
           'np_to_xshards',
           'xshard_to_np']


def np_to_data_creator(tuple_data):
    def data_creator(config, batch_size):
        data_len = tuple_data[0].shape[0]
        data = tf.data.Dataset.from_tensor_slices((tuple_data))
        data = data.cache().shuffle(data_len).batch(batch_size)
        return data.prefetch(tf.data.AUTOTUNE)
    return data_creator


def tsdata_to_data_creator(tf_data, shuffle=True):
    def data_creator(config, batch_size):
        data = tf_data.to_tf_dataset(shuffle=shuffle, batch_size=batch_size)
        return data
    return data_creator


def np_to_tfdataset(tuple_data, shuffle=True, batch_size=32):
    data = tf.data.Dataset.from_tensor_slices((tuple_data))
    if shuffle:
        data = data.cache().shuffle(tuple_data[0].shape[0]).batch(batch_size)
    else:
        data = data.batch(batch_size).cache()
    return data.prefetch(tf.data.AUTOTUNE)


def np_to_xshards(data):
    from bigdl.orca.data import XShards
    _shards = XShards.partition(data)

    def transform_to_dict(data):
        return {"x": data}
    return _shards.transform_shard(transform_to_dict)


def xshard_to_np(shard, mode="fit", expand_dim=None):
    if mode == "fit":
        data_local = shard.collect()
        return (np.concatenate([data_local[i]['x'] for i
                                in range(len(data_local))], axis=0),
                np.concatenate([data_local[i]['y'] for i
                                in range(len(data_local))], axis=0))
    if mode == "predict":
        data_local = shard.collect()
        return np.concatenate([data_local[i]['x'] for i
                               in range(len(data_local))], axis=0)
    if mode == "yhat":
        yhat = shard.collect()
        yhat = np.concatenate([yhat[i]['prediction'] for i in range(len(yhat))], axis=0)
        if len(expand_dim) >= 1:
            yhat = np.expand_dims(yhat, axis=expand_dim)
        return yhat
