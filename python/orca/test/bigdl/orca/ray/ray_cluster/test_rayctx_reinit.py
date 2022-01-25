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
import time
from unittest import TestCase

import numpy as np
import pytest
import ray

from bigdl.orca.ray import RayContext
from bigdl.orca.common import stop_orca_context

np.random.seed(1337)  # for reproducibility


@ray.remote
class TestRay():
    def hostname(self):
        import socket
        return socket.gethostname()


class TestUtil(TestCase):
    def setUp(self):
        self.ray_ctx = RayContext("ray", cores=2, num_nodes=1)
        self.node_num = 4
    
    def tearDown(self):
        stop_orca_context()

    def test_init_and_stop(self):
        self.ray_ctx.init()
        actors = [TestRay.remote() for i in range(0, self.node_num)]
        print(ray.get([actor.hostname.remote() for actor in actors]))
        self.ray_ctx.stop()
        assert not self.ray_ctx.initialized, ("The Ray cluster has been stopped.")
        time.sleep(3)
    
    def test_reinit(self):
        print("-------------------first repeat begin!------------------")
        self.ray_ctx = RayContext("ray", cores=2, num_nodes=1)
        assert RayContext._active_ray_context, ("Please create a RayContext First.")
        assert not self.ray_ctx.initialized, ("The Ray cluster has not been launched.")
        self.ray_ctx.init()
        actors = [TestRay.remote() for i in range(0, self.node_num)]
        print(ray.get([actor.hostname.remote() for actor in actors]))
        self.ray_ctx.stop()
        time.sleep(3)


if __name__ == "__main__":
    pytest.main([__file__])
