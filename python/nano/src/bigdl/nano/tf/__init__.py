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
import platform
import os
import warnings
from bigdl.nano.utils.common import register_suggestion
from bigdl.nano.utils.common import get_affinity_core_num


# reset the num of threads
if platform.system() == "Linux":
    # only UNIX-like system applied
    preset_thread_nums = int(os.environ.get("OMP_NUM_THREADS", "0"))
    affinity_core_num = get_affinity_core_num()

    if preset_thread_nums > affinity_core_num:
        register_suggestion(f"CPU Affinity is set to this program and {affinity_core_num} "
                            f"cores are binded. While Tensorflow OpenMP code block will use "
                            f"{preset_thread_nums} cores, which may cause severe performance "
                            f"downgrade. Please set `OMP_NUM_THREADS` to {affinity_core_num}.")


if platform.system() != "Darwin":
    # set tf settings
    import tensorflow as tf
    tf.config.set_soft_device_placement(enabled=True)

from .dispatcher import patch_tensorflow, unpatch_tensorflow
