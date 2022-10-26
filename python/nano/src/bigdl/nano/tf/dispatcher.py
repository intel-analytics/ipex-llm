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

from importlib.util import find_spec


_mapping_tf = None


def _get_patch_map():
    global _mapping_tf

    # decide if generate
    patch_tf = find_spec("tensorflow") is not None

    if patch_tf and _mapping_tf is None:
        _mapping_tf = []
        import keras
        import tensorflow
        from bigdl.nano.tf.keras import Sequential
        from bigdl.nano.tf.keras import Model
        from bigdl.nano.tf.optimizers import SparseAdam
        from bigdl.nano.tf.keras.layers import Embedding
        _mapping_tf += [
            [tensorflow.keras, "Model", Model, None],
            [tensorflow.keras, "Sequential", Sequential, None],
            [tensorflow.keras.optimizers, "Adam", SparseAdam, None],
            [tensorflow.keras.layers, "Embedding", Embedding, None],
            [keras, "Model", Model, None],
            [keras, "Sequential", Sequential, None],
            [keras.layers, "Embedding", Embedding, None]
        ]

    if not patch_tf:
        _mapping_tf = []

    return _mapping_tf


def patch_tensorflow():
    """
    patch_tensorflow is used to patch optimized tensorflow classes to replace original ones.

    Optimized classes include:

    | 1. tf.keras.Model/keras.Model -> bigdl.nano.tf.keras.Model
    | 2. tf.keras.Sequential/keras.Sequential -> bigdl.nano.tf.keras.Sequential
    | 3. tf.keras.layers.Embedding/keras.layers.Embedding -> bigdl.nano.tf.keras.layers.Embedding
    | 4. tf.optimizers.Adam -> bigdl.nano.tf.optimizers.SparseAdam
    """
    mapping_tf = _get_patch_map()

    for mapping_iter in mapping_tf:
        if mapping_iter[3] is None:
            mapping_iter[3] = getattr(mapping_iter[0], mapping_iter[1], None)
        setattr(mapping_iter[0], mapping_iter[1], mapping_iter[2])


def unpatch_tensorflow():
    """unpatch_tensorflow is used to unpatch optimized tensorflow classes to original ones."""
    mapping_tf = _get_patch_map()

    for mapping_iter in mapping_tf:
        setattr(mapping_iter[0], mapping_iter[1], mapping_iter[3])
