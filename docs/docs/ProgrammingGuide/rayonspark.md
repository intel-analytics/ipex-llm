---
### **Introduction**

[Ray](https://github.com/ray-project/ray) is a distributed framework for emerging AI applications open-sourced by [UC Berkeley RISELab](https://rise.cs.berkeley.edu). 
It implements a unified interface, distributed scheduler, and distributed and fault-tolerant store to address the new and demanding systems requirements for advanced AI technologies. 

Ray allows users to easily and efficiently to run many emerging AI applications, such as deep reinforcement learning using RLlib, scalable hyperparameter search using Ray Tune, automatic program synthesis using AutoPandas, etc.

Analytics Zoo provides a mechanism to deploy Python dependencies and Ray services automatically
across yarn cluster, meaning python users would be able to run `analytics-zoo` or `ray`
in a pythonic way on yarn without `spark-submit` or installing analytics-zoo or ray across all cluster nodes.

---
### **Steps to run RayOnSpark**

**NOTE:** We have been tested on Ray 0.8.4 and you are highly recommended to use this Ray version.

1) Install [Conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html) in your environment.

2) Create a new conda environment (with name "zoo" for example):
```
conda create -n zoo python=3.6
source activate zoo
```

3) Install analytics-zoo in the created conda environment:
```
pip install analytics-zoo[ray]
```

Note that the essential dependencies (including `ray==0.8.4`, `psutil`, `aiohttp`, `setproctitle`) will be installed by specifying the extras key `[ray]` when you pip install analytics-zoo.

4) Download JDK8 and set the environment variable: JAVA_HOME (recommended).

You can also install JDK via conda without setting the JAVA_HOME manually:

`conda install -c anaconda openjdk=8.0.152`

5) Start `python` and then execute the following example.

- Create a SparkContext on yarn, only __yarn-client mode__ is supported for now:

```python
from zoo import init_spark_on_yarn

sc = init_spark_on_yarn(
    hadoop_conf="path to the yarn configuration folder",
    conda_name="zoo", # The name of the created conda-env
    num_executor=2,
    executor_cores=4,
    executor_memory="8g",
    driver_memory="2g",
    driver_cores=4,
    extra_executor_memory_for_ray="10g")
```

- [Optional] If you don't have a yarn cluster, this can also be tested locally by creating a SparkContext
with `init_spark_on_local`:

```python
from zoo import init_spark_on_local

sc = init_spark_on_local(cores=4)
```

- Once the SparkContext is created, we can write more logic here such as training an Analytics Zoo model
or launching ray on Spark.

- Run the following simple example to launch a ray cluster on top of the SparkContext configurations and verify if RayOnSpark can work smoothly.

```python
import ray
from zoo.ray import RayContext

ray_ctx = RayContext(sc=sc, object_store_memory="5g")
ray_ctx.init()

@ray.remote
class TestRay():
    def hostname(self):
        import socket
        return socket.gethostname()

    def ip(self):
        import ray.services as rservices
        return rservices.get_node_ip_address()


actors = [TestRay.remote() for i in range(0, slave_num)]
print(ray.get([actor.hostname.remote() for actor in actors]))
print(ray.get([actor.ip.remote() for actor in actors]))
ray_ctx.stop()
```
