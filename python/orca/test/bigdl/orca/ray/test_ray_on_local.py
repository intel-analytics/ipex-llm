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
from unittest import TestCase

import pytest
import ray

from bigdl.dllib.nncontext import init_spark_on_local
from bigdl.orca.ray import RayContext


class TestRayLocal(TestCase):

    def test_local(self):
        @ray.remote
        class TestRay:
            def hostname(self):
                import socket
                return socket.gethostname()

        sc = init_spark_on_local(cores=8)
        config = {"object_spilling_config":"{\"type\":\"filesystem\","
                                           "\"params\":{\"directory_path\":\"/tmp/spill\"}}"}
        ray_ctx = RayContext(sc=sc, object_store_memory="1g", ray_node_cpu_cores=4,
                             system_config=config)
        address_info = ray_ctx.init()
        assert "object_store_address" in address_info
        actors = [TestRay.remote() for i in range(0, 4)]
        print(ray.get([actor.hostname.remote() for actor in actors]))
        ray_ctx.stop()
        sc.stop()


if __name__ == "__main__":
    pytest.main([__file__])
