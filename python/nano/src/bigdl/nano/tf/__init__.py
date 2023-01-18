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


if platform.system() != "Darwin":
    # set tf settings
    import tensorflow as tf
    if "PERF_MODE" in os.environ and os.environ["PERF_MODE"] == "1":
        tf.config.threading.set_inter_op_parallelism_threads(1)
        if "OMP_NUM_THREADS" in os.environ:
            tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["OMP_NUM_THREADS"]))

    tf.config.set_soft_device_placement(enabled=True)

from .dispatcher import patch_tensorflow, unpatch_tensorflow
