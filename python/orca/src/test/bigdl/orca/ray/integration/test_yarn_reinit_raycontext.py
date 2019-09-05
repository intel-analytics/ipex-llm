#
# Copyright 2018 Analytics Zoo Authors.
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
from unittest import TestCase

import numpy as np
import psutil
import pytest
import ray
import time

from zoo import init_spark_on_yarn
from zoo.ray.util.raycontext import RayContext

np.random.seed(1337)  # for reproducibility


@ray.remote
class TestRay():
    def hostname(self):
        import socket
        return socket.gethostname()


node_num = 4
sc = init_spark_on_yarn(
    hadoop_conf="/opt/work/hadoop-2.7.2/etc/hadoop/",
    conda_name="rayexample",
    num_executor=node_num,
    executor_cores=28,
    executor_memory="10g",
    driver_memory="2g",
    driver_cores=4,
    extra_executor_memory_for_ray="30g")
ray_ctx = RayContext(sc=sc, object_store_memory="2g")
ray_ctx.init()
actors = [TestRay.remote() for i in range(0, node_num)]
print([ray.get(actor.hostname.remote()) for actor in actors])
ray_ctx.stop()
# repeat
ray_ctx = RayContext(sc=sc, object_store_memory="1g")
ray_ctx.init()
actors = [TestRay.remote() for i in range(0, node_num)]
print([ray.get(actor.hostname.remote()) for actor in actors])
ray_ctx.stop()

sc.stop()
time.sleep(3)
