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


import tensorflow as tf
from bigdl.nano.common.cpu_schedule import schedule_workers, get_cpu_info

proc_list = schedule_workers(1)
_, get_socket = get_cpu_info()

num_sockets = len(set(get_socket.values()))
num_threads = len(proc_list[0]) // num_sockets

tf.config.threading.set_inter_op_parallelism_threads(num_sockets)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(enabled=True)
