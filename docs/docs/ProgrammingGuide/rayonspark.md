## Run Ray on Spark

AnalyticsZoo has already provided a mechanism to deploy Python dependencies and Ray services automatically
across yarn cluster,meaning python user would be able to run `Analytics-Zoo` or `Ray`
in a pythonic way on yarn without `spark-submit` or installing Analytics-Zoo or Ray across all cluster nodes.


## Here are the steps to run RayOnSpark:

1) You should install [Conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html) and create a conda-env named "ray36"

2) Install some essential dependencies in the conda env.

```
pip install analytics-zoo
pip install pyspark==2.4.0 # 2.4.3 is OK as well.
pip install ray==0.6.6
pip install conda-pack
pip install psutil
pip install aiohttp
pip install setproctitle
```

3) Download JDK8 and set the environment variable: JAVA_HOME (recommended).
   - You can also install JDK via conda without setting the JAVA_HOME manually:
   `conda install -c anaconda openjdk=8.0.152`

4) Start python and then execute the following example

- Create a SparkContext on Yarn

``` python
import ray

from zoo import init_spark_on_yarn
from zoo.ray.util.raycontext import RayContext

slave_num = 2

sc = init_spark_on_yarn(
    hadoop_conf="/opt/work/almaren-yarn-config/",
    conda_name="ray36",
    num_executor=slave_num,
    executor_cores=4,
    executor_memory="8g",
    driver_memory="2g",
    driver_cores=4,
    extra_executor_memory_for_ray="10g")
```

- [Optional] If you don't have a yarn cluster, this can also be test locally by creating `SparkContext`
with `init_spark_on_local`

```Python
from zoo import init_spark_on_local
sc = init_spark_on_local(cores=4)

```


- Once the SparkContext created, we can write more logic here either training Analytics-Zoo model
or launching ray on spark.

- The following code would launch a ray cluster on top of the SparkContext configuration and also verify with a simple Ray example.

```python

ray_ctx = RayContext(sc=sc,
                       object_store_memory="5g")
ray_ctx.init()


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


actors = [TestRay.remote() for i in range(0, slave_num)]
print([ray.get(actor.hostname.remote()) for actor in actors])
print([ray.get(actor.ip.remote()) for actor in actors])
ray_ctx.stop()

```

- NOTE: This was test on Ray 0.6.6. Ideally, we can upgrade to the latest version once the following issue is addressed.(https://github.com/ray-project/ray/issues/5223)


