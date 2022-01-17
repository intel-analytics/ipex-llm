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


class RayContext(object):

    _active_ray_context = None

    def __init__(self,
                 cluster_mode="ray_on_spark",
                 cores=None,
                 num_nodes=None,
                 **kwargs):

        self.cluster_mode = cluster_mode
        self.initialized = False

        if cluster_mode == "ray_on_spark":
            from bigdl.orca.ray.ray_on_spark_context import RayOnSparkContext
            self._ray_on_spark_context = RayOnSparkContext(**kwargs)
            self.is_local = self._ray_on_spark_context.is_local

        elif cluster_mode == "ray":
            self.is_local = False
            ray_args = kwargs.copy()
            self.ray_args = ray_args
        else:
            raise ValueError(f"Unsupported cluster mode: {cluster_mode}. "
                             f"Cluster mode must be ray or ray_on_spark")

        self.num_ray_nodes = num_nodes
        self.ray_node_cpu_cores = cores

        RayContext._active_ray_context = self

    def init(self, driver_cores=0):
        if self.cluster_mode == "ray":
            import ray
            results = ray.init(**self.ray_args)
        else:
            results = self._ray_on_spark_context.init(driver_cores=driver_cores)

        self.initialized = True
        return results

    def stop(self):
        if not self.initialized:
            print("The Ray cluster has not been launched.")
            return
        import ray
        ray.shutdown()
        self.initialized = False

    @classmethod
    def get(cls, initialize=True):
        if RayContext._active_ray_context:
            ray_ctx = RayContext._active_ray_context
            if initialize and not ray_ctx.initialized:
                ray_ctx.init()
            return ray_ctx
        else:
            raise Exception("No active RayContext. Please create a RayContext and init it first")
