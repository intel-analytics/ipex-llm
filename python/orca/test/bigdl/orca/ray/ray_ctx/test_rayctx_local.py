#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#

#
from unittest import TestCase

import pytest
import ray

from bigdl.orca import stop_orca_context
from bigdl.orca.ray import RayContext

@ray.remote
class TestRay():
    def hostname(self):
        import socket
        return socket.gethostname()

class TestUtil(TestCase):
    def setUp(self):
        self.ray_ctx = RayContext("ray", cores=2, num_nodes=1)
    
    def tearDown(self):
        stop_orca_context()

    def test_init(self):
        node_num = 4
        address_info = self.ray_ctx.init()
        assert RayContext._active_ray_context, ("RayContext has not been initialized")
        assert "object_store_address" in address_info
        actors = [TestRay.remote() for i in range(0, node_num)]
        print(ray.get([actor.hostname.remote() for actor in actors]))
    
    def test_stop(self):
        self.ray_ctx.stop()
        assert not self.ray_ctx.initialized, ("The Ray cluster has been stopped.")

    def test_get(self):
        self.ray_ctx.get()
        assert self.ray_ctx.initialized, ("The Ray cluster has been launched.")

if __name__ == "__main__":
    pytest.main([__file__])
