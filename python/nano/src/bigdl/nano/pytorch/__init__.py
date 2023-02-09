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


if 'KMP_INIT_AT_FORK' in os.environ:
    del os.environ['KMP_INIT_AT_FORK']

# reset the num of threads
if platform.system() == "Linux":
    # only UNIX-like system applied
    preset_thread_nums = torch.get_num_threads()

    # When using proclist to bind cores,
    # `os.sched_getaffinity(0)` will return only the first bound core,
    # so we need to parse KMP_AFFINITY manually in this case
    KMP_AFFINITY = os.environ.get("KMP_AFFINITY", "")
    if "proclist" not in KMP_AFFINITY:
        affinity_core_num = len(os.sched_getaffinity(0))
    else:
        try:
            start_pos = KMP_AFFINITY.find('[', KMP_AFFINITY.find("proclist")) + 1
            end_pos = KMP_AFFINITY.find(']', start_pos)
            proclist_str = KMP_AFFINITY[start_pos:end_pos].replace(" ", "")
            core_list = []
            for n in proclist_str.split(','):
                if '-' not in n:
                    core_list.append(int(n))
                else:
                    start, end = n.split('-')
                    core_list.extend(range(int(start), int(end) + 1))
            affinity_core_num = len(core_list)
        except Exception as _e:
            warnings.warn(f"Failed to parse KMP_AFFINITY: '{KMP_AFFINITY}'."
                          f" Will use default thread number: {preset_thread_nums}")
            affinity_core_num = preset_thread_nums

    if preset_thread_nums > affinity_core_num:
        register_suggestion(f"CPU Affinity is set to this program and {affinity_core_num} "
                            f"cores are binded. While OpenMP code block will use "
                            f"{preset_thread_nums} cores, which may cause severe performance "
                            f"downgrade. Please set `OMP_NUM_THREADS` to {affinity_core_num}.")

from .dispatcher import patch_torch, unpatch_torch
from bigdl.nano.pytorch.inference import InferenceOptimizer
from bigdl.nano.pytorch.trainer import Trainer
from bigdl.nano.pytorch.torch_nano import TorchNano
from bigdl.nano.pytorch.torch_nano import nano
