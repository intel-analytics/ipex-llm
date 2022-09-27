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

def patch_nano(patch_tf=None, patch_torch=None):
    '''
    This patching function is used to patch optimized class to replace original ones.
    Optimized classes include:
    1. tf.keras.Model/keras.Model -> bigdl.nano.tf.keras.Model
    2. tf.keras.Model/keras.Sequential -> bigdl.nano.tf.keras.Sequential
    3. tf.keras.layers.Embedding/keras.layers.Embedding -> bigdl.nano.tf.keras.layers.Embedding
    4. tf.optimizers.SparseAdam -> bigdl.nano.tf.optimizers.SparseAdam
    5. pytorch_lightning.Trainer -> bigdl.nano.pytorch.Trainer

    :param patch_tf: bool, if patch tensorflow related classes, will patch defaultly if tensorflow
           is installed
    :param patch_torch: bool, if patch pytorch related classes, will patch defaultly if pytorch
           is installed
    '''
    if find_spec("tensorflow") is not None and patch_tf is None:
        patch_tf = True
    if find_spec("pytorch_lightning") is not None and patch_torch is None:
        patch_torch = True

    if patch_tf:
        import keras
        import tensorflow as tf
        from bigdl.nano.tf.keras import Sequential
        from bigdl.nano.tf.keras import Model
        from bigdl.nano.tf.optimizers import SparseAdam
        from bigdl.nano.tf.keras.layers import Embedding

        setattr(tf.keras, "Model", Model)
        setattr(tf.keras, "Sequential", Sequential)
        setattr(tf.keras.layers, "Embedding", Embedding)
        setattr(keras, "Model", Model)
        setattr(keras, "Sequential", Sequential)
        setattr(keras.layers, "Embedding", Embedding)
        setattr(tf.optimizers, "Adam", SparseAdam)

    if patch_torch:
        import pytorch_lightning
        from bigdl.nano.pytorch import Trainer

        setattr(pytorch_lightning, "Trainer", Trainer)
