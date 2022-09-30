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
import os
from logging import warning
from importlib.util import find_spec
from functools import lru_cache


envs_checklist = ["LD_PRELOAD", "OMP_NUM_THREADS", "KMP_AFFINITY",
                  "KMP_BLOCKTIME"]

_unset_envs = []


def _check_nano_envs():
    for k in envs_checklist:
        if not os.environ.get(k, None):
            _unset_envs.append(k)

    if len(_unset_envs):
        highlight_boundary = "\n{}\n".format("*" * 150)
        warning(f"{highlight_boundary}Nano environment variables {_unset_envs} are not set.\n"
                f"Please run `source bigdl-nano-init` to initialize them, "
                f"or you can prefix `bigdl-nano-init` to the command you run.\n"
                f"\nExample:\n"
                f"bigdl-nano-init python pytorch-lenet.py --device ipex"
                f"{highlight_boundary}")

# disable env check for now, as it does not work for tf and windows
# _check_nano_envs()


@lru_cache(maxsize=None)
def get_patch_map():

    patch_tf = find_spec("tensorflow") is not None
    patch_torch = find_spec("pytorch_lightning") is not None
    mapping_tf = []
    mapping_torch = []

    # 4-dim list, where, what, which, legancy
    if patch_tf:
        import keras
        import tensorflow as tf
        from bigdl.nano.tf.keras import Sequential
        from bigdl.nano.tf.keras import Model
        from bigdl.nano.tf.optimizers import SparseAdam
        from bigdl.nano.tf.keras.layers import Embedding
        mapping_tf += [
                        [tf.keras, "Model", Model, None],
                        [tf.keras, "Sequential", Sequential, None],
                        [tf.keras.layers, "Embedding", Embedding, None],
                        [keras, "Model", Model, None],
                        [keras, "Sequential", Sequential, None],
                        [keras.layers, "Embedding", Embedding, None],
                        [tf.optimizers, "Adam", SparseAdam, None]
                      ]

    if patch_torch:
        import pytorch_lightning
        import torchvision
        from bigdl.nano.pytorch import Trainer
        from bigdl.nano.pytorch.vision import transforms
        from bigdl.nano.pytorch.vision import datasets
        mapping_torch += [
                            [pytorch_lightning, "Trainer", Trainer, None],
                            [torchvision, "transforms", transforms, None],
                            [torchvision, "datasets", datasets, None],
                         ]

    return mapping_tf, mapping_torch


def patch_nano(patch_tf=None, patch_torch=None):
    '''
    This patching function is used to patch optimized class to replace original ones.
    Optimized classes include:
    1. tf.keras.Model/keras.Model -> bigdl.nano.tf.keras.Model
    2. tf.keras.Model/keras.Sequential -> bigdl.nano.tf.keras.Sequential
    3. tf.keras.layers.Embedding/keras.layers.Embedding -> bigdl.nano.tf.keras.layers.Embedding
    4. tf.optimizers.Adam -> bigdl.nano.tf.optimizers.SparseAdam
    5. pytorch_lightning.Trainer -> bigdl.nano.pytorch.Trainer
    6. torchvision.transforms -> bigdl.nano.pytorch.vision.transforms
    7. torchvision.datasets -> bigdl.nano.pytorch.vision.datasets

    :param patch_tf: bool, if patch tensorflow related classes, will patch defaultly if tensorflow
           is installed
    :param patch_torch: bool, if patch pytorch related classes, will patch defaultly if pytorch
           is installed
    '''
    if patch_tf is None:
        patch_tf = find_spec("tensorflow") is not None
    if patch_torch is None:
        patch_torch = find_spec("pytorch_lightning") is not None

    mapping_tf, mapping_torch = get_patch_map()

    if patch_tf:
        for mapping_iter in mapping_tf:
            mapping_iter[3] = getattr(mapping_iter[0], mapping_iter[1], None)
            setattr(mapping_iter[0], mapping_iter[1], mapping_iter[2])
    
    if patch_torch:
        for mapping_iter in mapping_torch:
            mapping_iter[3] = getattr(mapping_iter[0], mapping_iter[1], None)
            setattr(mapping_iter[0], mapping_iter[1], mapping_iter[2])


def unpatch_nano(patch_tf=None, patch_torch=None):
    '''
    This unpatching function is used to unpatch optimized class to original ones.

    :param patch_tf: bool, if patch tensorflow related classes, will patch defaultly if tensorflow
           is installed
    :param patch_torch: bool, if patch pytorch related classes, will patch defaultly if pytorch
           is installed
    '''
    if patch_tf is None:
        patch_tf = find_spec("tensorflow") is not None
    if patch_torch is None:
        patch_torch = find_spec("pytorch_lightning") is not None

    mapping_tf, mapping_torch = get_patch_map()

    if patch_tf:
        for mapping_iter in mapping_tf:
            setattr(mapping_iter[0], mapping_iter[1], mapping_iter[3])
    
    if patch_torch:
        for mapping_iter in mapping_torch:
            setattr(mapping_iter[0], mapping_iter[1], mapping_iter[3])
