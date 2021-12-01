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


    def __init__(self, cluster_mode="ray_on_spark", **kwargs):

        self.cluster_mode = cluster_mode

        if cluster_mode == "ray_on_spark":
            from bigdl.orca.ray.ray_on_spark_context import RayOnSparkContext
            self._ray_on_spark_context = RayOnSparkContext(**kwargs)
            self.num_ray_nodes = self._ray_on_spark_context.num_ray_nodes
            self.ray_node_cpu_cores = self._ray_on_spark_context.ray_node_cpu_cores
            self.is_local = self._ray_on_spark_context.is_local
        
        elif cluster_mode == "ray":
            assert "num_ray_nodes" in kwargs, ("num_ray_nodes must be specific" +
                                               " when cluster_mode is ray")
            assert "ray_node_cpu_cores" in kwargs, ("ray_node_cpu_cores must be specific" +
                                                  " when cluster_mode is ray")
            self.num_ray_nodes = kwargs["num_ray_nodes"]
            self.ray_node_cpu_cores = kwargs["ray_node_cpu_cores"]
            self.is_local = False
            ray_args = kwargs.copy()
            del ray_args["num_ray_nodes"]
            del ray_args["ray_node_cpu_cores"]
            self.ray_args = ray_args

    def init(self):
        if self.cluster_mode == "ray":
            import ray
            ray.init(**self.ray_args)
        else:
            self._ray_on_spark_context.init()

    def stop(self):
        pass

    @classmethod
    def get()