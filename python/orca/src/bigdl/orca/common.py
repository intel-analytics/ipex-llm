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

import os
from bigdl.dllib.nncontext import ZooContext


class OrcaContextMeta(type):

    _pandas_read_backend = "spark"
    __eager_mode = True
    _serialize_data_creator = False
    _train_data_store = "DRAM"
    __shard_size = None

    @property
    def log_output(cls):
        """
        Whether to redirect Spark driver JVM's stdout and stderr to the current
        python process. This is useful when running Analytics Zoo in jupyter notebook.
        Default to be False. Needs to be set before initializing SparkContext.
        """
        return ZooContext.log_output

    @log_output.setter
    def log_output(cls, value):
        ZooContext.log_output = value

    @property
    def pandas_read_backend(cls):
        """
        The backend for reading csv/json files. Either "spark" or "pandas".
        spark backend would call spark.read and pandas backend would call pandas.read.
        Default to be "spark".
        """
        return cls._pandas_read_backend

    @pandas_read_backend.setter
    def pandas_read_backend(cls, value):
        value = value.lower()
        assert value == "spark" or value == "pandas", \
            "pandas_read_backend must be either spark or pandas"
        cls._pandas_read_backend = value

    @property
    def _eager_mode(cls):
        """
        Whether to compute eagerly for SparkXShards.
        Default to be True.
        """
        return cls.__eager_mode

    @_eager_mode.setter
    def _eager_mode(cls, value):
        assert isinstance(value, bool), "_eager_mode should either be True or False"
        cls.__eager_mode = value

    @property
    def serialize_data_creator(cls):
        """
        Whether add a file lock to the data loading process for PyTorch Horovod training.
        This would be useful when you run multiple workers on a single node to download data
        to the same destination.
        Default to be False.
        """
        return cls._serialize_data_creator

    @serialize_data_creator.setter
    def serialize_data_creator(cls, value):
        assert isinstance(value, bool), "serialize_data_creator should either be True or False"
        cls._serialize_data_creator = value

    @property
    def train_data_store(cls):
        """
        The memory type for train data storage. Either 'DRAM', 'PMEM', or 'DISK_n'.
        The default value is 'DRAM', you can change it to 'PMEM' if have AEP hardware.
        If you give 'DISK_n', in which 'n' is an integer, we will cache the data into disk,
        and hold only `1/n` of the data in memory. After going through the `1/n`,
        we will release the current cache, and load another `1/n` into memory.
        """
        return cls._train_data_store

    @train_data_store.setter
    def train_data_store(cls, value):
        value = value.upper()
        import re
        assert value == "DRAM" or value == "PMEM" or re.match("DISK_\d+", value), \
            "train_data_store must be either DRAM or PMEM or DIRECT or DISK_n"
        cls._train_data_store = value

    @property
    def _shard_size(cls):
        """
        The number of Rows in Spark DataFrame to transform as one shard of SparkXShards. We convert
        Spark DataFrame input to SparkXShards internally in fit/predict/evaluate of
        PyTorchRayEstimator and TensorFlow2Estimator. This parameter may affect the performance in
        transferring an SparkXShards to an RayXShards.
        Default to be None, in which case Rows in one partition will be transformed as one shard.
        """
        return cls.__shard_size

    @_shard_size.setter
    def _shard_size(cls, value):
        if value is not None:
            assert isinstance(value, int) and value > 0, \
                "shard size should be either None or a positive integer."
        cls.__shard_size = value

    @property
    def barrier_mode(cls):
        """
        Whether to use Spark barrier mode to launch Ray, which is supported in Spark 2.4+ and when
        dynamic allocation is disabled.
        Default to be True.
        """
        return ZooContext.barrier_mode

    @barrier_mode.setter
    def barrier_mode(cls, value):
        ZooContext.barrier_mode = value


class OrcaContext(metaclass=OrcaContextMeta):
    @staticmethod
    def get_spark_context():
        from pyspark import SparkContext
        if SparkContext._active_spark_context is not None:
            return SparkContext.getOrCreate()
        else:
            raise Exception("No active SparkContext. Please create a SparkContext first")

    @staticmethod
    def get_sql_context():
        from pyspark.sql import SQLContext
        return SQLContext.getOrCreate(OrcaContext.get_spark_context())

    @staticmethod
    def get_spark_session():
        return OrcaContext.get_sql_context().sparkSession

    @staticmethod
    def get_ray_context():
        from bigdl.orca.ray import RayContext
        return RayContext.get()


def _check_python_micro_version():
    # with ray >=1.8.0, python small micro version will cause pickle error in ray.init()
    # (https://github.com/ray-project/ray/issues/19938)
    import sys
    if sys.version_info[2] < 3:
        raise RuntimeError(f"Found python version {sys.version[:5]}. We only support python"
                           f"with micro version >= 10 (e.g. 3.{sys.version_info[1]}.10)")


def init_orca_context(cluster_mode=None, cores=2, memory="2g", num_nodes=1,
                      init_ray_on_spark=False, **kwargs):
    """
    Creates or gets a SparkContext for different Spark cluster modes (and launch Ray services
    across the cluster if necessary).

    :param cluster_mode: The mode for the Spark cluster. One of "local", "yarn-client",
           "yarn-cluster", "k8s-client" and "standalone". Default to be None and in this case
           there is supposed to be an existing SparkContext in your application.

           For "yarn-client" and "yarn-cluster", you are supposed to use conda environment
           and set the environment variable HADOOP_CONF_DIR.

           For "k8s-client", you are supposed to additionally specify the arguments master
           and container_image.
           For "k8s-cluster", you are supposed to use spark-submit to submit the application
           and use the default cluster_mode instead.
           In this case, please set the Spark configurations through command line options or
           the properties file.
           To make things easier, you are recommended to use the launch scripts we provide:
           https://github.com/intel-analytics/BigDL/tree/branch-2.0/scripts.

           For other cluster modes, you are recommended to install and run bigdl through
           pip, which is more convenient.
    :param cores: The number of cores to be used on each node. Default to be 2.
    :param memory: The memory allocated for each node. Default to be '2g'.
    :param num_nodes: The number of nodes to be used in the cluster. Default to be 1.
           For Spark local, num_nodes should always be 1 and you don't need to change it.
    :param init_ray_on_spark: Whether to launch Ray services across the cluster.
           Default to be False and in this case the Ray cluster would be launched lazily when
           Ray is involved in Project Orca.
    :param kwargs: The extra keyword arguments used for creating SparkContext and
           launching Ray if any.

    :return: An instance of SparkContext.
    """
    print("Initializing orca context")
    import atexit
    atexit.register(stop_orca_context)
    from pyspark import SparkContext
    import warnings
    spark_args = {}
    for key in ["conf", "spark_log_level", "redirect_spark_log"]:
        if key in kwargs:
            spark_args[key] = kwargs[key]
    if cluster_mode is not None:
        cluster_mode = cluster_mode.lower()
    activate_sc = SparkContext._active_spark_context is not None
    if activate_sc:
        if cluster_mode is not None and cluster_mode != "spark-submit":
            warnings.warn("Use an existing SparkContext, " +
                          "cluster_mode is determined by the existing SparkContext", Warning)
        from bigdl.dllib.nncontext import init_nncontext
        sc = init_nncontext(conf=None, spark_log_level="WARN", redirect_spark_log=True)
    else:
        cluster_mode = "local" if cluster_mode is None else cluster_mode
        if cluster_mode == "local":
            if num_nodes > 1:
                warnings.warn("For Spark local mode, num_nodes should be 1, but got "
                              + repr(num_nodes) + ", ignored", Warning)
            os.environ["SPARK_DRIVER_MEMORY"] = memory
            if "python_location" in kwargs:
                spark_args["python_location"] = kwargs["python_location"]
            from bigdl.dllib.nncontext import init_spark_on_local
            sc = init_spark_on_local(cores, **spark_args)
        elif cluster_mode == "spark-submit":
            from bigdl.dllib.nncontext import init_nncontext
            sc = init_nncontext(**spark_args)
        elif cluster_mode.startswith("yarn"):  # yarn, yarn-client or yarn-cluster
            hadoop_conf = os.environ.get("HADOOP_CONF_DIR")
            if not hadoop_conf:
                assert "hadoop_conf" in kwargs,\
                    "Directory path to hadoop conf not found for yarn-client mode. Please either " \
                    "specify argument hadoop_conf or set the environment variable HADOOP_CONF_DIR"
                hadoop_conf = kwargs["hadoop_conf"]
            from bigdl.dllib.utils.utils import detect_conda_env_name
            conda_env_name = detect_conda_env_name()
            for key in ["driver_cores", "driver_memory", "extra_executor_memory_for_ray",
                        "extra_python_lib", "penv_archive", "additional_archive",
                        "hadoop_user_name", "spark_yarn_archive", "jars", "py_files"]:
                if key in kwargs:
                    spark_args[key] = kwargs[key]
            from bigdl.dllib.nncontext import init_spark_on_yarn, init_spark_on_yarn_cluster
            if cluster_mode == "yarn-cluster":
                sc = init_spark_on_yarn_cluster(hadoop_conf=hadoop_conf,
                                                conda_name=conda_env_name,
                                                num_executors=num_nodes,
                                                executor_cores=cores,
                                                executor_memory=memory,
                                                **spark_args)
            else:
                sc = init_spark_on_yarn(hadoop_conf=hadoop_conf,
                                        conda_name=conda_env_name,
                                        num_executors=num_nodes, executor_cores=cores,
                                        executor_memory=memory, **spark_args)
        elif cluster_mode.startswith("k8s"):  # k8s or k8s-client
            if cluster_mode == "k8s-cluster":
                raise ValueError('For k8s-cluster mode, '
                                 'please submit the application via spark-submit'
                                 'and use the default cluster_mode instead')
            assert "master" in kwargs, "Please specify master for k8s-client mode"
            assert "container_image" in kwargs, "Please specify container_image for k8s-client mode"
            from bigdl.dllib.utils.utils import detect_conda_env_name
            conda_env_name = detect_conda_env_name()
            for key in ["driver_cores", "driver_memory", "extra_executor_memory_for_ray",
                        "extra_python_lib", "penv_archive", "jars", "python_location"]:
                if key in kwargs:
                    spark_args[key] = kwargs[key]
            from bigdl.dllib.nncontext import init_spark_on_k8s
            sc = init_spark_on_k8s(master=kwargs["master"],
                                   container_image=kwargs["container_image"],
                                   conda_name=conda_env_name,
                                   num_executors=num_nodes, executor_cores=cores,
                                   executor_memory=memory, **spark_args)
        elif cluster_mode == "standalone":
            for key in ["driver_cores", "driver_memory", "extra_executor_memory_for_ray",
                        "extra_python_lib", "jars", "master", "python_location",
                        "enable_numa_binding"]:
                if key in kwargs:
                    spark_args[key] = kwargs[key]
            from bigdl.dllib.nncontext import init_spark_standalone
            sc = init_spark_standalone(num_executors=num_nodes, executor_cores=cores,
                                       executor_memory=memory, **spark_args)
        else:
            raise ValueError("cluster_mode can only be local, yarn-client, yarn-cluster,"
                             "k8s-client or standalone, "
                             "but got: %s".format(cluster_mode))
    ray_args = {}
    for key in ["redis_port", "password", "object_store_memory", "verbose", "env",
                "extra_params", "num_ray_nodes", "ray_node_cpu_cores", "include_webui"]:
        if key in kwargs:
            ray_args[key] = kwargs[key]
    from bigdl.orca.ray import RayContext
    ray_ctx = RayContext(sc, **ray_args)
    if init_ray_on_spark:
        _check_python_micro_version()
        driver_cores = 0  # This is the default value.
        ray_ctx.init(driver_cores=driver_cores)
    return sc


def stop_orca_context():
    """
    Stop the SparkContext (and stop Ray services across the cluster if necessary).
    """
    from pyspark import SparkContext
    # If users successfully call stop_orca_context after the program finishes,
    # namely when there is no active SparkContext, the registered exit function
    # should do nothing.
    if SparkContext._active_spark_context is not None:
        print("Stopping orca context")
        from bigdl.orca.ray import RayContext
        ray_ctx = RayContext.get(initialize=False)
        if ray_ctx.initialized:
            ray_ctx.stop()
        sc = SparkContext.getOrCreate()
        if sc.getConf().get("spark.master").startswith("spark://"):
            from bigdl.dllib.nncontext import stop_spark_standalone
            stop_spark_standalone()
        sc.stop()
