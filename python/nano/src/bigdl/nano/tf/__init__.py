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
    # get socket number and phy core per socket
    from bigdl.nano.common.cpu_schedule import get_cpu_info
    l_core_to_p_core, l_core_to_socket = get_cpu_info()
    physical_core = int(max(l_core_to_p_core.values()) + 1)
    socket_num = int(max(l_core_to_socket.values()) + 1)
    physical_core_per_socket = int(physical_core // socket_num)

    # set tf settings
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(socket_num)
    tf.config.threading.set_intra_op_parallelism_threads(physical_core_per_socket)
    tf.config.set_soft_device_placement(enabled=True)

from .dispatcher import patch_tensorflow, unpatch_tensorflow
