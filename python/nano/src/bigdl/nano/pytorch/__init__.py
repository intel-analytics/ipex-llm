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
__all__ = ["Trainer", "TorchNano", "InferenceOptimizer"]


# unset the KMP_INIT_AT_FORK
# which will cause significant slow down in multiprocessing training
import os
import torch
import platform
import warnings
from bigdl.nano.utils.common import register_suggestion
from bigdl.nano.utils.common import get_affinity_core_num


# fix schedule when using GOMP
ld_preload = os.environ.get("LD_PRELOAD", "")
if "libiomp5.so" not in ld_preload:
    torch.set_num_threads(torch.get_num_threads())

if 'KMP_INIT_AT_FORK' in os.environ:
    del os.environ['KMP_INIT_AT_FORK']

# reset the num of threads
if platform.system() == "Linux":
    # only UNIX-like system applied
    preset_thread_nums = torch.get_num_threads()
    affinity_core_num = get_affinity_core_num()

    if affinity_core_num is not None and preset_thread_nums > affinity_core_num:
        register_suggestion(f"CPU Affinity is set to this program and {affinity_core_num} "
                            f"cores are binded. While PyTorch OpenMP code block will use "
                            f"{preset_thread_nums} cores, which may cause severe performance "
                            f"downgrade. Please set `OMP_NUM_THREADS` to {affinity_core_num}.")

from .dispatcher import patch_torch, unpatch_torch
from bigdl.nano.pytorch.inference import InferenceOptimizer
from bigdl.nano.pytorch.inference import Pipeline
from bigdl.nano.pytorch.trainer import Trainer
from bigdl.nano.pytorch.torch_nano import TorchNano
from bigdl.nano.pytorch.torch_nano import nano
