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

from tempfile import TemporaryDirectory

import tensorflow as tf
from tensorflow.keras import mixed_precision

from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.tf import patch_compiled


def BF16Model(model, custom_objects=None):
    original_policies = []
    policy_bf16 = mixed_precision.Policy('mixed_bfloat16')
    for layer in model.layers:
        original_policies.append(layer._dtype_policy)
        layer._set_dtype_policy(policy_bf16)
    try:
        # save or load operation may fail
        with TemporaryDirectory() as temp_dir:
            model.save(temp_dir)
            bf16_model = tf.keras.models.load_model(temp_dir, custom_objects=custom_objects)
    except Exception as _e:
        # if a functional model failed to save or load, try `from_config`
        if hasattr(model, "input_shape"):
            config = model.get_config()
            bf16_model = tf.keras.Model.from_config(config)
            with TemporaryDirectory() as temp_dir:
                model.save_weights(temp_dir)
                bf16_model.load_weights(temp_dir)
            patch_compiled(bf16_model, model)
        else:
            invalidInputError(False, "Failed to convert model to bfloat16 dtype.")
    finally:
        for policy, layer in zip(original_policies, model.layers):
            layer._set_dtype_policy(policy)
    return bf16_model


def load_bf16_model(path):
    return tf.keras.models.load_model(path)
