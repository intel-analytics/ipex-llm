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


import os
import re
import subprocess
import time
import uuid
import random
import warnings
import tempfile
import filelock
import multiprocessing
from packaging import version

from zoo.ray.process import session_execute, ProcessMonitor
from zoo.ray.utils import is_local
from zoo.ray.utils import resource_to_bytes
from zoo.ray.utils import get_parent_pid


def kill_redundant_log_monitors(redis_address):

    """
    Killing redundant log_monitor.py processes.
    If multiple ray nodes are started on the same machine,
    there will be multiple ray log_monitor.py processes
    monitoring the same log dir. As a result, the logs
    will be replicated multiple times and forwarded to driver.
    See issue https://github.com/ray-project/ray/issues/10392
    """

    import psutil
    import subprocess
    log_monitor_processes = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            # Avoid throw exception when listing lwsslauncher in macOS
            if proc.name() is None or proc.name() == "lwsslauncher":
                continue
            cmdline = subprocess.list2cmdline(proc.cmdline())
            is_log_monitor = "log_monitor.py" in cmdline
            is_same_redis = "--redis-address={}".format(redis_address)
            if is_log_monitor and is_same_redis in cmdline:
                log_monitor_processes.append(proc)
        except (psutil.AccessDenied, psutil.ZombieProcess, psutil.ProcessLookupError):
            # psutil may encounter AccessDenied or ZombieProcess exceptions
            # when it's trying to visit some MacOS core services
            if psutil.MACOS:
                continue
            else:
                raise Exception("List process with list2cmdline failed!")

    if len(log_monitor_processes) > 1:
        for proc in log_monitor_processes[1:]:
            proc.kill()


class RayServiceFuncGenerator(object):
    """
    This should be a pickable class.
    """
    def _prepare_env(self):
        modified_env = os.environ.copy()
        if self.python_loc == "python_env/bin/python":
            # In this case the executor is using the conda yarn archive under the current
            # working directory. Need to get the full path.
            executor_python_path = "{}/{}".format(
                os.getcwd(), "/".join(self.python_loc.split("/")[:-1]))
        else:
            executor_python_path = "/".join(self.python_loc.split("/")[:-1])
        if "PATH" in os.environ:
            modified_env["PATH"] = "{}:{}".format(executor_python_path, os.environ["PATH"])
        else:
            modified_env["PATH"] = executor_python_path
        modified_env.pop("MALLOC_ARENA_MAX", None)
        modified_env.pop("RAY_BACKEND_LOG_LEVEL", None)
        # Unset all MKL setting as Analytics Zoo would give default values when init env.
        # Running different programs may need different configurations.
        modified_env.pop("intra_op_parallelism_threads", None)
        modified_env.pop("inter_op_parallelism_threads", None)
        modified_env.pop("OMP_NUM_THREADS", None)
        modified_env.pop("KMP_BLOCKTIME", None)
        modified_env.pop("KMP_AFFINITY", None)
        modified_env.pop("KMP_SETTINGS", None)
        if self.env:  # Add in env argument if any MKL setting is needed.
            modified_env.update(self.env)
        if self.verbose:
            print("Executing with these environment settings:")
            for pair in modified_env.items():
                print(pair)
            print("The $PATH is: {}".format(modified_env["PATH"]))
        return modified_env

    def __init__(self, python_loc, redis_port, ray_node_cpu_cores,
                 password, object_store_memory, verbose=False, env=None,
                 include_webui=False,
                 extra_params=None):
        """object_store_memory: integer in bytes"""
        self.env = env
        self.python_loc = python_loc
        self.redis_port = redis_port
        self.password = password
        self.ray_node_cpu_cores = ray_node_cpu_cores
        self.ray_exec = self._get_ray_exec()
        self.object_store_memory = object_store_memory
        self.extra_params = extra_params
        self.include_webui = include_webui
        self.verbose = verbose
        # _mxnet_worker and _mxnet_server are resource tags for distributed MXNet training only
        # in order to diff worker from server.
        # This is useful to allocate workers and servers in the cluster.
        # Leave some reserved custom resources free to avoid unknown crash due to resources.
        self.labels = \
            """--resources '{"_mxnet_worker": %s, "_mxnet_server": %s, "_reserved": %s}'""" \
            % (1, 1, 2)
        # Add a unique id so that different Ray programs won't affect each other even if
        # the flags and locks are not removed.
        tag = uuid.uuid4().hex
        self.ray_master_flag = "ray_master_{}".format(tag)
        self.ray_master_lock = "ray_master_start_{}.lock".format(tag)
        self.raylet_lock = "raylet_start_{}.lock".format(tag)

    def gen_stop(self):
        def _stop(iter):
            command = "{} stop".format(self.ray_exec)
            print("Start to end the ray services: {}".format(command))
            session_execute(command=command, fail_fast=True)
            return iter

        return _stop

    @staticmethod
    def _enrich_command(command, object_store_memory, extra_params):
        if object_store_memory:
            command = command + " --object-store-memory {}".format(str(object_store_memory))
        if extra_params:
            for pair in extra_params.items():
                command = command + " --{} {}".format(pair[0], pair[1])
        return command

    def _gen_master_command(self):
        webui = "true" if self.include_webui else "false"
        command = "{} start --head " \
                  "--include-dashboard {} --dashboard-host 0.0.0.0 --port {} " \
                  "--redis-password {} --num-cpus {}". \
            format(self.ray_exec, webui, self.redis_port, self.password,
                   self.ray_node_cpu_cores)
        if self.labels:
            command = command + " " + self.labels
        return RayServiceFuncGenerator._enrich_command(command=command,
                                                       object_store_memory=self.object_store_memory,
                                                       extra_params=self.extra_params)

    @staticmethod
    def _get_raylet_command(redis_address,
                            ray_exec,
                            password,
                            ray_node_cpu_cores,
                            labels="",
                            object_store_memory=None,
                            extra_params=None):
        command = "{} start --address {} --redis-password {} --num-cpus {}".format(
            ray_exec, redis_address, password, ray_node_cpu_cores)
        if labels:
            command = command + " " + labels
        return RayServiceFuncGenerator._enrich_command(command=command,
                                                       object_store_memory=object_store_memory,
                                                       extra_params=extra_params)

    @staticmethod
    def _get_spark_executor_pid():
        # TODO: This might not work on OS other than Linux
        this_pid = os.getpid()
        pyspark_daemon_pid = get_parent_pid(this_pid)
        spark_executor_pid = get_parent_pid(pyspark_daemon_pid)
        return spark_executor_pid

    @staticmethod
    def start_ray_daemon(python_loc, pid_to_watch, pgid_to_kill):
        daemon_path = os.path.join(os.path.dirname(__file__), "ray_daemon.py")
        start_daemon_command = ['nohup', python_loc, daemon_path, str(pid_to_watch),
                                str(pgid_to_kill)]
        # put ray daemon process in its children's process group to avoid being killed by spark.
        subprocess.Popen(start_daemon_command, preexec_fn=os.setpgrp)
        time.sleep(1)

    def _start_ray_node(self, command, tag):
        modified_env = self._prepare_env()
        print("Starting {} by running: {}".format(tag, command))
        process_info = session_execute(command=command, env=modified_env, tag=tag)
        spark_executor_pid = RayServiceFuncGenerator._get_spark_executor_pid()
        RayServiceFuncGenerator.start_ray_daemon(self.python_loc,
                                                 pid_to_watch=spark_executor_pid,
                                                 pgid_to_kill=process_info.pgid)
        import ray._private.services as rservices
        process_info.node_ip = rservices.get_node_ip_address()
        return process_info

    def _get_ray_exec(self):
        if "envs" in self.python_loc:  # conda environment
            python_bin_dir = "/".join(self.python_loc.split("/")[:-1])
            return "{}/python {}/ray".format(python_bin_dir, python_bin_dir)
        elif self.python_loc == "python_env/bin/python":  # conda yarn archive on the executor
            return "python_env/bin/python python_env/bin/ray"
        else:  # system environment with ray installed; for example: /usr/local/bin/ray
            return "ray"

    def gen_ray_master_start(self):
        def _start_ray_master(index, iter):
            from zoo.util.utils import get_node_ip
            process_info = None
            if index == 0:
                print("partition id is : {}".format(index))
                current_ip = get_node_ip()
                print("master address {}".format(current_ip))
                redis_address = "{}:{}".format(current_ip, self.redis_port)
                process_info = self._start_ray_node(command=self._gen_master_command(),
                                                    tag="ray-master")
                process_info.master_addr = redis_address
            yield process_info
        return _start_ray_master

    def gen_raylet_start(self, redis_address):
        def _start_raylets(iter):
            from zoo.util.utils import get_node_ip
            current_ip = get_node_ip()
            master_ip = redis_address.split(":")[0]
            do_start = True
            process_info = None
            base_path = tempfile.gettempdir()
            ray_master_flag_path = os.path.join(base_path, self.ray_master_flag)
            # If there is already a ray master on this node, we need to start one less raylet.
            if current_ip == master_ip:
                ray_master_lock_path = os.path.join(base_path, self.ray_master_lock)
                with filelock.FileLock(ray_master_lock_path):
                    if not os.path.exists(ray_master_flag_path):
                        os.mknod(ray_master_flag_path)
                        do_start = False
            if do_start:
                raylet_lock_path = os.path.join(base_path, self.raylet_lock)
                with filelock.FileLock(raylet_lock_path):
                    process_info = self._start_ray_node(
                        command=RayServiceFuncGenerator._get_raylet_command(
                            redis_address=redis_address,
                            ray_exec=self.ray_exec,
                            password=self.password,
                            ray_node_cpu_cores=self.ray_node_cpu_cores,
                            labels=self.labels,
                            object_store_memory=self.object_store_memory,
                            extra_params=self.extra_params),
                        tag="raylet")
                    kill_redundant_log_monitors(redis_address=redis_address)
            # Cannot remove ray_master_flag at the end of this task since no barrier is guaranteed.

            yield process_info
        return _start_raylets

    def gen_ray_start(self, master_ip):
        def _start_ray_services(iter):
            from pyspark import BarrierTaskContext
            from zoo.util.utils import get_node_ip
            tc = BarrierTaskContext.get()
            current_ip = get_node_ip()
            print("current address {}".format(current_ip))
            print("master address {}".format(master_ip))
            redis_address = "{}:{}".format(master_ip, self.redis_port)
            process_info = None
            base_path = tempfile.gettempdir()
            ray_master_flag_path = os.path.join(base_path, self.ray_master_flag)
            if current_ip == master_ip:  # Start the ray master.
                # It is possible that multiple executors are on one node. In this case,
                # the first executor that gets the lock would be the master and it would
                # create a flag to indicate the master has initialized.
                # The flag file is removed when ray start processes finish so that this
                # won't affect other programs.
                ray_master_lock_path = os.path.join(base_path, self.ray_master_lock)
                with filelock.FileLock(ray_master_lock_path):
                    if not os.path.exists(ray_master_flag_path):
                        print("partition id is : {}".format(tc.partitionId()))
                        process_info = self._start_ray_node(command=self._gen_master_command(),
                                                            tag="ray-master")
                        process_info.master_addr = redis_address
                        os.mknod(ray_master_flag_path)

            tc.barrier()
            if not process_info:  # Start raylets.
                # Add a lock to avoid starting multiple raylets on one node at the same time.
                # See this issue: https://github.com/ray-project/ray/issues/10154
                raylet_lock_path = os.path.join(base_path, self.raylet_lock)
                with filelock.FileLock(raylet_lock_path):
                    print("partition id is : {}".format(tc.partitionId()))
                    process_info = self._start_ray_node(
                        command=RayServiceFuncGenerator._get_raylet_command(
                            redis_address=redis_address,
                            ray_exec=self.ray_exec,
                            password=self.password,
                            ray_node_cpu_cores=self.ray_node_cpu_cores,
                            labels=self.labels,
                            object_store_memory=self.object_store_memory,
                            extra_params=self.extra_params),
                        tag="raylet")
                    kill_redundant_log_monitors(redis_address=redis_address)

            if os.path.exists(ray_master_flag_path):
                os.remove(ray_master_flag_path)
            yield process_info
        return _start_ray_services


class RayContext(object):
    _active_ray_context = None

    def __init__(self, sc, redis_port=None, password="123456", object_store_memory=None,
                 verbose=False, env=None, extra_params=None, include_webui=True,
                 num_ray_nodes=None, ray_node_cpu_cores=None):
        """
        The RayContext would initiate a ray cluster on top of the configuration of SparkContext.
        After creating RayContext, call the init method to set up the cluster.

        - For Spark local mode: The total available cores for Ray is equal to the number of
        Spark local cores.
        - For Spark cluster mode: The number of raylets to be created is equal to the number of
        Spark executors. The number of cores allocated for each raylet is equal to the number of
        cores for each Spark executor.
        You are allowed to specify num_ray_nodes and ray_node_cpu_cores for configurations
        to start raylets.

        :param sc: An instance of SparkContext.
        :param redis_port: The redis port for the ray head node. Default is None.
        The value would be randomly picked if not specified.
        :param password: The password for redis. Default to be "123456" if not specified.
        :param object_store_memory: The memory size for ray object_store in string.
        This can be specified in bytes(b), kilobytes(k), megabytes(m) or gigabytes(g).
        For example, "50b", "100k", "250m", "30g".
        :param verbose: True for more logs when starting ray. Default is False.
        :param env: The environment variable dict for running ray processes. Default is None.
        :param extra_params: The key value dict for extra options to launch ray.
        For example, extra_params={"temp-dir": "/tmp/ray/"}
        :param include_webui: True for including web ui when starting ray. Default is False.
        :param num_ray_nodes: The number of raylets to start across the cluster.
        For Spark local mode, you don't need to specify this value.
        For Spark cluster mode, it is default to be the number of Spark executors. If
        spark.executor.instances can't be detected in your SparkContext, you need to explicitly
        specify this. It is recommended that num_ray_nodes is not larger than the number of
        Spark executors to make sure there are enough resources in your cluster.
        :param ray_node_cpu_cores: The number of available cores for each raylet.
        For Spark local mode, it is default to be the number of Spark local cores.
        For Spark cluster mode, it is default to be the number of cores for each Spark executor. If
        spark.executor.cores or spark.cores.max can't be detected in your SparkContext, you need to
        explicitly specify this. It is recommended that ray_node_cpu_cores is not larger than the
        number of cores for each Spark executor to make sure there are enough resources in your
        cluster.
        """
        assert sc is not None, "sc cannot be None, please create a SparkContext first"
        self.sc = sc
        self.initialized = False
        self.is_local = is_local(sc)
        self.verbose = verbose
        self.redis_password = password
        self.object_store_memory = resource_to_bytes(object_store_memory)
        self.ray_processesMonitor = None
        self.env = env
        self.extra_params = extra_params
        self.include_webui = include_webui
        self._address_info = None
        if self.is_local:
            self.num_ray_nodes = 1
            spark_cores = self._get_spark_local_cores()
            if ray_node_cpu_cores:
                ray_node_cpu_cores = int(ray_node_cpu_cores)
                if ray_node_cpu_cores > spark_cores:
                    warnings.warn("ray_node_cpu_cores is larger than available Spark cores, "
                                  "make sure there are enough resources on your machine")
                self.ray_node_cpu_cores = ray_node_cpu_cores
            else:
                self.ray_node_cpu_cores = spark_cores
        # For Spark local mode, directly call ray.init() and ray.shutdown().
        # ray.shutdown() would clear up all the ray related processes.
        # Ray Manager is only needed for Spark cluster mode to monitor ray processes.
        else:
            if self.sc.getConf().contains("spark.executor.cores"):
                executor_cores = int(self.sc.getConf().get("spark.executor.cores"))
            else:
                executor_cores = None
            if ray_node_cpu_cores:
                ray_node_cpu_cores = int(ray_node_cpu_cores)
                if executor_cores and ray_node_cpu_cores > executor_cores:
                    warnings.warn("ray_node_cpu_cores is larger than Spark executor cores, "
                                  "make sure there are enough resources on your cluster")
                self.ray_node_cpu_cores = ray_node_cpu_cores
            elif executor_cores:
                self.ray_node_cpu_cores = executor_cores
            else:
                raise Exception("spark.executor.cores not detected in the SparkContext, "
                                "you need to manually specify num_ray_nodes and ray_node_cpu_cores "
                                "for RayContext to start ray services")
            if self.sc.getConf().contains("spark.executor.instances"):
                num_executors = int(self.sc.getConf().get("spark.executor.instances"))
            elif self.sc.getConf().contains("spark.cores.max"):
                import math
                num_executors = math.floor(
                    int(self.sc.getConf().get("spark.cores.max")) / self.ray_node_cpu_cores)
            else:
                num_executors = None
            if num_ray_nodes:
                num_ray_nodes = int(num_ray_nodes)
                if num_executors and num_ray_nodes > num_executors:
                    warnings.warn("num_ray_nodes is larger than the number of Spark executors, "
                                  "make sure there are enough resources on your cluster")
                self.num_ray_nodes = num_ray_nodes
            elif num_executors:
                self.num_ray_nodes = num_executors
            else:
                raise Exception("spark.executor.cores not detected in the SparkContext, "
                                "you need to manually specify num_ray_nodes and ray_node_cpu_cores "
                                "for RayContext to start ray services")

            from zoo.util.utils import detect_python_location
            self.python_loc = os.environ.get("PYSPARK_PYTHON", detect_python_location())
            self.redis_port = random.randint(10000, 65535) if not redis_port else int(redis_port)
            self.ray_service = RayServiceFuncGenerator(
                python_loc=self.python_loc,
                redis_port=self.redis_port,
                ray_node_cpu_cores=self.ray_node_cpu_cores,
                password=self.redis_password,
                object_store_memory=self.object_store_memory,
                verbose=self.verbose,
                env=self.env,
                include_webui=self.include_webui,
                extra_params=self.extra_params)
        RayContext._active_ray_context = self
        self.total_cores = self.num_ray_nodes * self.ray_node_cpu_cores

    @classmethod
    def get(cls, initialize=True):
        if RayContext._active_ray_context:
            ray_ctx = RayContext._active_ray_context
            if initialize and not ray_ctx.initialized:
                ray_ctx.init()
            return ray_ctx
        else:
            raise Exception("No active RayContext. Please create a RayContext and init it first")

    def _gather_cluster_ips(self):
        """
        Get the ips of all Spark executors in the cluster. The first ip returned would be the
        ray master.
        """
        def info_fn(iter):
            from zoo.util.utils import get_node_ip
            yield get_node_ip()

        ips = self.sc.range(0, self.total_cores,
                            numSlices=self.total_cores).mapPartitions(info_fn).collect()
        ips = list(set(ips))
        return ips

    def stop(self):
        if not self.initialized:
            print("The Ray cluster has not been launched.")
            return
        import ray
        ray.shutdown()
        self.initialized = False

    def purge(self):
        """
        Invoke ray stop to clean ray processes.
        """
        if not self.initialized:
            print("The Ray cluster has not been launched.")
            return
        if self.is_local:
            import ray
            ray.shutdown()
        else:
            self.sc.range(0, self.total_cores,
                          numSlices=self.total_cores).mapPartitions(
                self.ray_service.gen_stop()).collect()
        self.initialized = False

    def _get_spark_local_cores(self):
        local_symbol = re.match(r"local\[(.*)\]", self.sc.master).group(1)
        if local_symbol == "*":
            return multiprocessing.cpu_count()
        else:
            return int(local_symbol)

    def init(self, driver_cores=0):
        """
        Initiate the ray cluster.

        :param driver_cores: The number of cores for the raylet on driver for Spark cluster mode.
        Default is 0 and in this case the local driver wouldn't have any ray workload.

        :return The dictionary of address information about the ray cluster.
        Information contains node_ip_address, redis_address, object_store_address,
        raylet_socket_name, webui_url and session_dir.
        """
        if self.initialized:
            print("The Ray cluster has been launched.")
        else:
            if self.is_local:
                if self.env:
                    os.environ.update(self.env)
                import ray
                kwargs = {}
                if self.extra_params is not None:
                    for k, v in self.extra_params.items():
                        kw = k.replace("-", "_")
                        kwargs[kw] = v
                init_params = dict(
                    num_cpus=self.ray_node_cpu_cores,
                    _redis_password=self.redis_password,
                    object_store_memory=self.object_store_memory,
                    include_dashboard=self.include_webui,
                    dashboard_host="0.0.0.0",
                )
                init_params.update(kwargs)
                if version.parse(ray.__version__) >= version.parse("1.4.0"):
                    init_params["namespace"] = "az"
                self._address_info = ray.init(**init_params)
            else:
                self.cluster_ips = self._gather_cluster_ips()
                redis_address = self._start_cluster()
                self._address_info = self._start_driver(num_cores=driver_cores,
                                                        redis_address=redis_address)

            print(self._address_info)
            kill_redundant_log_monitors(self._address_info["redis_address"])
            self.initialized = True
        return self._address_info

    @property
    def address_info(self):
        if self._address_info:
            return self._address_info
        else:
            raise Exception("The Ray cluster has not been launched yet. Please call init first")

    @property
    def redis_address(self):
        return self.address_info["redis_address"]

    def _start_cluster(self):
        ray_rdd = self.sc.range(0, self.num_ray_nodes,
                                numSlices=self.num_ray_nodes)
        from zoo import ZooContext
        if ZooContext.barrier_mode:
            print("Launching Ray on cluster with Spark barrier mode")
            # The first ip would be used to launch ray master.
            process_infos = ray_rdd.barrier().mapPartitions(
                self.ray_service.gen_ray_start(self.cluster_ips[0])).collect()
        else:
            print("Launching Ray on cluster without Spark barrier mode")
            master_process_infos = ray_rdd.mapPartitionsWithIndex(
                self.ray_service.gen_ray_master_start()).collect()
            master_process_infos = [process for process in master_process_infos if process]
            assert len(master_process_infos) == 1, \
                "There should be only one ray master launched, but got {}"\
                .format(len(master_process_infos))
            master_process_info = master_process_infos[0]
            redis_address = master_process_info.master_addr
            raylet_process_infos = ray_rdd.mapPartitions(
                self.ray_service.gen_raylet_start(redis_address)).collect()
            raylet_process_infos = [process for process in raylet_process_infos if process]
            assert len(raylet_process_infos) == self.num_ray_nodes - 1, \
                "There should be {} raylets launched across the cluster, but got {}"\
                .format(self.num_ray_nodes - 1, len(raylet_process_infos))
            process_infos = master_process_infos + raylet_process_infos

        self.ray_processesMonitor = ProcessMonitor(process_infos, self.sc, ray_rdd, self,
                                                   verbose=self.verbose)
        return self.ray_processesMonitor.master.master_addr

    def _start_restricted_worker(self, num_cores, node_ip_address, redis_address):
        extra_param = {"node-ip-address": node_ip_address}
        if self.extra_params is not None:
            extra_param.update(self.extra_params)
        command = RayServiceFuncGenerator._get_raylet_command(
            redis_address=redis_address,
            ray_exec="ray",
            password=self.redis_password,
            ray_node_cpu_cores=num_cores,
            object_store_memory=self.object_store_memory,
            extra_params=extra_param)
        modified_env = self.ray_service._prepare_env()
        print("Executing command: {}".format(command))
        process_info = session_execute(command=command, env=modified_env,
                                       tag="raylet", fail_fast=True)
        RayServiceFuncGenerator.start_ray_daemon("python",
                                                 pid_to_watch=os.getpid(),
                                                 pgid_to_kill=process_info.pgid)

    def _start_driver(self, num_cores, redis_address):
        print("Start to launch ray driver on local")
        import ray._private.services
        node_ip = ray._private.services.get_node_ip_address(redis_address)
        self._start_restricted_worker(num_cores=num_cores,
                                      node_ip_address=node_ip,
                                      redis_address=redis_address)
        ray.shutdown()
        init_params = dict(
            address=redis_address,
            _redis_password=self.ray_service.password,
            _node_ip_address=node_ip
        )
        if version.parse(ray.__version__) >= version.parse("1.4.0"):
            init_params["namespace"] = "az"
        return ray.init(**init_params)
