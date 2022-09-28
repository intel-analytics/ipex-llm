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


from threading import Lock

from bigdl.dllib.utils.log4Error import invalidInputError


class OrcaRayContext(object):

    _active_ray_context = None
    _lock = Lock()

    def __init__(self,
                 runtime="spark",
                 cores=2,
                 num_nodes=1,
                 **kwargs):
        # Add sys.stdout.fileno for Databricks. In Databricks notebook, sys.stdout is redirected to
        # a ConsoleBuffer object, this object has no attribute fileno, and cause ray init crash.
        # Normally, sys.stdout should have attribute fileno and is set to 1.
        # So set sys.stdout.fileno to 1 when this attribute is missing.
        import sys
        if not hasattr(sys.stdout, 'fileno'):
            sys.stdout.fileno = lambda: 1

        self.runtime = runtime
        self.initialized = False

        if runtime == "spark":
            from bigdl.orca.ray import RayOnSparkContext
            self._ray_on_spark_context = RayOnSparkContext(**kwargs)
            self.is_local = self._ray_on_spark_context.is_local

        elif runtime == "ray":
            self.is_local = False
            ray_args = kwargs.copy()
            self.ray_args = ray_args
            self.num_ray_nodes = num_nodes
            self.ray_node_cpu_cores = cores
        else:
            invalidInputError(False,
                              f"Unsupported runtime: {runtime}. "
                              f"Runtime must be spark or ray")

        OrcaRayContext._active_ray_context = self

    def init(self, driver_cores=0):
        if self.runtime == "ray":
            import ray
            results = ray.init(**self.ray_args)
        else:
            results = self._ray_on_spark_context.init(driver_cores=driver_cores)
            self.num_ray_nodes = self._ray_on_spark_context.num_ray_nodes
            self.ray_node_cpu_cores = self._ray_on_spark_context.ray_node_cpu_cores
            self.address_info = self._ray_on_spark_context.address_info
            self.redis_address = self._ray_on_spark_context.redis_address
            self.redis_password = self._ray_on_spark_context.redis_password
            self.sc = self._ray_on_spark_context.sc

        self.initialized = True
        return results

    def stop(self):
        if not self.initialized:
            print("The Ray cluster has not been launched.")
            return
        import ray
        ray.shutdown()
        self.initialized = False
        with OrcaRayContext._lock:
            OrcaRayContext._active_ray_context = None

    @classmethod
    def get(cls, initialize=True):
        if OrcaRayContext._active_ray_context:
            ray_ctx = OrcaRayContext._active_ray_context
            if initialize and not ray_ctx.initialized:
                ray_ctx.init()
            return ray_ctx
        else:
            invalidInputError(False,
                              "No active RayContext. "
                              "Please call init_orca_context to create a RayContext.")
