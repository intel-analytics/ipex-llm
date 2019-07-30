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


import ray

from zoo import init_spark_on_yarn
from zoo.ray.util.raycontext import RayContext

slave_num = 2

sc = init_spark_on_yarn(
    hadoop_conf="/opt/work/almaren-yarn-config/",
    conda_name="ray_train",
    num_executor=slave_num,
    executor_cores=28,
    executor_memory="10g",
    driver_memory="2g",
    driver_cores=4,
    extra_executor_memory_for_ray="30g",
    spark_conf={"hello": "world"})

ray_ctx = RayContext(sc=sc,
                     object_store_memory="25g",
                     extra_params={"temp-dir": "/tmp/hello/"},
                     env={"http_proxy": "http://child-prc.intel.com:913",
                          "http_proxys": "http://child-prc.intel.com:913"})
ray_ctx.init(object_store_memory="2g",
             num_cores=0,
             labels="",
             extra_params={})


@ray.remote
class TestRay():
    def hostname(self):
        import socket
        return socket.gethostname()

    def check_cv2(self):
        # conda install -c conda-forge opencv==3.4.2
        import cv2
        return cv2.__version__

    def ip(self):
        import ray.services as rservices
        return rservices.get_node_ip_address()

    def network(self):
        from urllib.request import urlopen
        try:
            urlopen('http://www.baidu.com', timeout=3)
            return True
        except Exception as err:
            return False


actors = [TestRay.remote() for i in range(0, slave_num)]
print([ray.get(actor.hostname.remote()) for actor in actors])
print([ray.get(actor.ip.remote()) for actor in actors])
# print([ray.get(actor.network.remote()) for actor in actors])

ray_ctx.stop()
