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

import multiprocessing
import os
import random
import re
import signal

from pyspark import BarrierTaskContext
from zoo.ray.process import session_execute, ProcessMonitor
from zoo.ray.utils import is_local
from zoo.ray.utils import resource_to_bytes


class JVMGuard:
    """
    The registered pids would be put into the killing list of Spark Executor.
    """

    @staticmethod
    def register_pids(pids):
        import traceback
        try:
            from zoo.common.utils import callZooFunc
            import zoo
            callZooFunc("float",
                        "jvmGuardRegisterPids",
                        pids)
        except Exception as err:
            print(traceback.format_exc())
            print("Cannot sucessfully register pid into JVMGuard")
            for pid in pids:
                os.kill(pid, signal.SIGKILL)
            raise err


class RayServiceFuncGenerator(object):
    """
    This should be a pickable class.
    """
    def _prepare_env(self):
        modified_env = os.environ.copy()
        cwd = os.getcwd()
        modified_env["PATH"] = "{}/{}:{}".format(cwd, "/".join(self.python_loc.split("/")[:-1]),
                                                 os.environ["PATH"])
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
            print("Executing with these environment setting:")
            for pair in modified_env.items():
                print(pair)
            print("The $PATH is: {}".format(modified_env["PATH"]))
        return modified_env

    def __init__(self, python_loc, redis_port, ray_node_cpu_cores,
                 password, object_store_memory, verbose=False, env=None,
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
        self.verbose = verbose
        # _mxnet_worker and _mxnet_server are resource tags for distributed MXNet training only
        # in order to diff worker from server.
        # This is useful to allocate workers and servers in the cluster.
        # Leave some reserved custom resources free to avoid unknown crash due to resources.
        self.labels = \
            """--resources='{"_mxnet_worker": %s, "_mxnet_server": %s, "_reserved": %s}' """ \
            % (1, 1, 2)

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
            command = command + "--object-store-memory {} ".format(str(object_store_memory))
        if extra_params:
            for pair in extra_params.items():
                command = command + " --{} {} ".format(pair[0], pair[1])
        return command

    def _gen_master_command(self):
        command = "{} start --head " \
                  "--include-webui --redis-port {} " \
                  "--redis-password {} --num-cpus {} {}". \
            format(self.ray_exec, self.redis_port, self.password,
                   self.ray_node_cpu_cores, self.labels)
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
        command = "{} start --redis-address {} --redis-password  {} --num-cpus {} {}  ".format(
            ray_exec, redis_address, password, ray_node_cpu_cores, labels)
        return RayServiceFuncGenerator._enrich_command(command=command,
                                                       object_store_memory=object_store_memory,
                                                       extra_params=extra_params)

    def _start_ray_node(self, command, tag):
        modified_env = self._prepare_env()
        print("Starting {} by running: {}".format(tag, command))
        process_info = session_execute(command=command, env=modified_env, tag=tag)
        JVMGuard.register_pids(process_info.pids)
        import ray.services as rservices
        process_info.node_ip = rservices.get_node_ip_address()
        return process_info

    def _get_ray_exec(self):
        python_bin_dir = "/".join(self.python_loc.split("/")[:-1])
        return "{}/python {}/ray".format(python_bin_dir, python_bin_dir)

    def gen_ray_start(self):
        def _start_ray_services(iter):
            tc = BarrierTaskContext.get()
            # The address is sorted by partitionId according to the comments
            # Partition 0 is the Master
            task_addrs = [taskInfo.address for taskInfo in tc.getTaskInfos()]
            print(task_addrs)
            master_ip = task_addrs[0].split(":")[0]
            print("current address {}".format(task_addrs[tc.partitionId()]))
            print("master address {}".format(master_ip))
            redis_address = "{}:{}".format(master_ip, self.redis_port)
            process_info = None
            if tc.partitionId() == 0:
                print("partition id is : {}".format(tc.partitionId()))
                process_info = self._start_ray_node(command=self._gen_master_command(),
                                                    tag="ray-master")
                process_info.master_addr = redis_address

            tc.barrier()
            if tc.partitionId() != 0:
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
            yield process_info
        return _start_ray_services


class RayContext(object):
    def __init__(self, sc, redis_port=None, password="123456", object_store_memory=None,
                 verbose=False, env=None, extra_params=None):
        """
        The RayContext would init a ray cluster on top of the configuration of SparkContext.
        For spark cluster mode: The number of raylets is equal to number of executors.
        For Spark local mode: The number of raylets is controlled by local_ray_node_num.
        CPU cores for each is raylet equals to spark_cores/local_ray_node_num.
        :param sc:
        :param redis_port: redis port for the "head" node.
               The value would be randomly picked if not specified.
        :param password: [optional] password for the redis.
        :param object_store_memory: Memory size for the object_store.
        :param verbose: True for more logs.
        :param env: The environment variable dict for running Ray.
        :param extra_params: key value dictionary for extra options to launch Ray.
                             i.e extra_params={"temp-dir": "/tmp/ray2/"}
        """
        assert sc is not None, "sc cannot be None, please create a SparkContext first"
        self.sc = sc
        self.stopped = False
        self.is_local = is_local(sc)
        self.verbose = verbose
        self.redis_password = password
        self.object_store_memory = resource_to_bytes(object_store_memory)
        self.ray_processesMonitor = None
        self.env = env
        self.extra_params = extra_params
        if self.is_local:
            self.num_ray_nodes = 1
            self.ray_node_cpu_cores = self._get_spark_local_cores()
        # For Spark local mode, directly call ray.init() and ray.shutdown().
        # ray.shutdown() would clear up all the ray related processes.
        # Ray Manager is only needed for Spark cluster mode to monitor ray processes.
        else:
            self.num_ray_nodes = int(self.sc.getConf().get("spark.executor.instances"))
            self.ray_node_cpu_cores = int(self.sc.getConf().get("spark.executor.cores"))
            self.python_loc = os.environ['PYSPARK_PYTHON']
            self.redis_port = random.randint(10000, 65535) if not redis_port else redis_port
            self.ray_service = RayServiceFuncGenerator(
                python_loc=self.python_loc,
                redis_port=self.redis_port,
                ray_node_cpu_cores=self.ray_node_cpu_cores,
                password=self.redis_password,
                object_store_memory=self.object_store_memory,
                verbose=self.verbose,
                env=self.env,
                extra_params=self.extra_params)
            self._gather_cluster_ips()
            from bigdl.util.common import init_executor_gateway
            print("Start to launch the JVM guarding process")
            init_executor_gateway(sc)
            print("JVM guarding process has been successfully launched")

    def _gather_cluster_ips(self):
        total_cores = int(self.num_ray_nodes) * int(self.ray_node_cpu_cores)

        def info_fn(iter):
            tc = BarrierTaskContext.get()
            task_addrs = [taskInfo.address.split(":")[0] for taskInfo in tc.getTaskInfos()]
            yield task_addrs
            tc.barrier()

        ips = self.sc.range(0, total_cores,
                            numSlices=total_cores).barrier().mapPartitions(info_fn).collect()
        return ips[0]

    def stop(self):
        if self.stopped:
            print("This instance has been stopped.")
            return
        import ray
        ray.shutdown()
        if not self.is_local:
            if not self.ray_processesMonitor:
                print("Please start the runner first before closing it")
            else:
                self.ray_processesMonitor.clean_fn()
        self.stopped = True

    def purge(self):
        """
        Invoke ray stop to clean ray processes
        """
        if self.stopped:
            print("This instance has been stopped.")
            return
        if self.is_local:
            import ray
            ray.shutdown()
        else:
            self.sc.range(0,
                          self.num_ray_nodes,
                          numSlices=self.num_ray_nodes).barrier().mapPartitions(
                self.ray_service.gen_stop()).collect()
        self.stopped = True

    def _get_spark_local_cores(self):
        local_symbol = re.match(r"local\[(.*)\]", self.sc.master).group(1)
        if local_symbol == "*":
            return multiprocessing.cpu_count()
        else:
            return int(local_symbol)

    def init(self):
        self.stopped = False
        if self.is_local:
            if self.env:
                os.environ.update(self.env)
            import ray
            ray.init(num_cpus=self.ray_node_cpu_cores,
                     object_store_memory=self.object_store_memory,
                     resources=self.extra_params)
        else:
            self._start_cluster()
            self._start_driver()

    def _start_cluster(self):
        print("Start to launch ray on cluster")
        ray_rdd = self.sc.range(0, self.num_ray_nodes,
                                numSlices=self.num_ray_nodes)
        process_infos = ray_rdd.barrier().mapPartitions(
            self.ray_service.gen_ray_start()).collect()

        self.ray_processesMonitor = ProcessMonitor(process_infos, self.sc, ray_rdd, self,
                                                   verbose=self.verbose)
        self.redis_address = self.ray_processesMonitor.master.master_addr
        return self

    def _start_restricted_worker(self, num_cores=0):
        command = RayServiceFuncGenerator._get_raylet_command(
            redis_address=self.redis_address,
            ray_exec="ray ",
            password=self.redis_password,
            ray_node_cpu_cores=num_cores,
            object_store_memory=self.object_store_memory,
            extra_params=self.extra_params)
        print("Executing command: {}".format(command))
        process_info = session_execute(command=command, fail_fast=True)
        ProcessMonitor.register_shutdown_hook(pgid=process_info.pgid)

    def _start_driver(self, num_cores=0):
        print("Start to launch ray driver on local")
        import ray
        if not self.is_local:
            self._start_restricted_worker(num_cores=num_cores)
        ray.shutdown()
        ray.init(redis_address=self.redis_address,
                 redis_password=self.ray_service.password)
