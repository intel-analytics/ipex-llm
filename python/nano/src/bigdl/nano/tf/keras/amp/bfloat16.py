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
from tensorflow.keras import mixed_precision
from tempfile import TemporaryDirectory
import tensorflow as tf


def BF16Model(model):
    # TODO: check bf16 isa
    policy_bf16 = mixed_precision.Policy('mixed_bfloat16')
    for layer in model.layers:
        layer._dtype_policy = policy_bf16
    with TemporaryDirectory() as temp_dir:
        model.save(temp_dir)
        bf16_model = tf.keras.models.load_model(temp_dir)
    return bf16_model


def load_bf16_model(path):
    return tf.keras.models.load_model(path)
