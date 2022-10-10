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

from functools import lru_cache
from importlib.util import find_spec


_mapping_tf = None
_mapping_torch = None


def _get_patch_map():
    global _mapping_tf
    global _mapping_torch

    patch_tf = find_spec("tensorflow") is not None
    patch_torch = find_spec("pytorch_lightning") is not None

    # 4-dim list, where, what, which, legancy
    if _mapping_tf is None:
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

    if _mapping_torch is None:
        _mapping_torch = []
        import pytorch_lightning
        import torchvision
        from bigdl.nano.pytorch import Trainer
        from bigdl.nano.pytorch.vision import transforms
        from bigdl.nano.pytorch.vision import datasets
        _mapping_torch += [
            [pytorch_lightning, "Trainer", Trainer, None],
            [torchvision, "transforms", transforms, None],
            [torchvision, "datasets", datasets, None],
        ]

    if not patch_tf:
        return [], _mapping_torch
    if not patch_torch:
        return _mapping_tf, []
    return _mapping_tf, _mapping_torch


def patch_nano(patch_tf=None, patch_torch=None):
    """

    patch_nano is used to patch optimized class to replace original ones.

    Optimized classes include:

    | 1. tf.keras.Model/keras.Model -> bigdl.nano.tf.keras.Model
    | 2. tf.keras.Sequential/keras.Sequential -> bigdl.nano.tf.keras.Sequential
    | 3. tf.keras.layers.Embedding/keras.layers.Embedding -> bigdl.nano.tf.keras.layers.Embedding
    | 4. tf.optimizers.Adam -> bigdl.nano.tf.optimizers.SparseAdam
    | 5. pytorch_lightning.Trainer -> bigdl.nano.pytorch.Trainer
    | 6. torchvision.transforms -> bigdl.nano.pytorch.vision.transforms
    | 7. torchvision.datasets -> bigdl.nano.pytorch.vision.datasets

    :param patch_tf: bool, if patch tensorflow related classes, will patch defaultly if tensorflow
           is installed
    :param patch_torch: bool, if patch pytorch related classes, will patch defaultly if pytorch
           is installed
    """
    if patch_tf is None:
        patch_tf = find_spec("tensorflow") is not None
    if patch_torch is None:
        patch_torch = find_spec("pytorch_lightning") is not None

    mapping_tf, mapping_torch = _get_patch_map()

    if patch_tf:
        for mapping_iter in mapping_tf:
            mapping_iter[3] = getattr(mapping_iter[0], mapping_iter[1], None)
            setattr(mapping_iter[0], mapping_iter[1], mapping_iter[2])

    if patch_torch:
        for mapping_iter in mapping_torch:
            mapping_iter[3] = getattr(mapping_iter[0], mapping_iter[1], None)
            setattr(mapping_iter[0], mapping_iter[1], mapping_iter[2])


def unpatch_nano(unpatch_tf=None, unpatch_torch=None):
    """
    unpatch_nano is used to unpatch optimized class to original ones.

    :param unpatch_tf: bool, if unpatch tensorflow related classes,
           will unpatch defaultly if tensorflow is installed
    :param unpatch_torch: bool, if unpatch pytorch related classes,
           will unpatch defaultly if pytorch is installed
    """
    if unpatch_tf is None:
        unpatch_tf = find_spec("tensorflow") is not None
    if unpatch_torch is None:
        unpatch_torch = find_spec("pytorch_lightning") is not None

    mapping_tf, mapping_torch = _get_patch_map()

    if unpatch_tf:
        for mapping_iter in mapping_tf:
            setattr(mapping_iter[0], mapping_iter[1], mapping_iter[3])

    if unpatch_torch:
        for mapping_iter in mapping_torch:
            setattr(mapping_iter[0], mapping_iter[1], mapping_iter[3])
