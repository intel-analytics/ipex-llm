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
import sys
import random
import platform

from pyspark import SparkContext
from bigdl.dllib.nncontext import init_internal_nncontext, init_spark_conf
from bigdl.dllib.utils.utils import detect_python_location, pack_penv, get_node_ip
from bigdl.dllib.utils.utils import get_executor_conda_zoo_classpath
from bigdl.dllib.utils.utils import get_zoo_bigdl_classpath_on_driver
from bigdl.dllib.utils.utils import get_bigdl_class_version
from bigdl.dllib.utils.utils import get_bigdl_image_workdir
from bigdl.dllib.utils.engine import get_bigdl_jars
from bigdl.dllib.utils.log4Error import *


class SparkRunner:
    standalone_env = None

    def __init__(self,
                 spark_log_level="WARN",
                 redirect_spark_log=True):
        self.spark_log_level = spark_log_level
        self.redirect_spark_log = redirect_spark_log
        with SparkContext._lock:
            if SparkContext._active_spark_context:
                print("WARNING: Note that there's an existing SparkContext. "
                      "Using the existing SparkContext.")
        import pyspark
        print("Current pyspark location is : {}".format(pyspark.__file__))

    def create_sc(self, submit_args, conf):
        submit_args = submit_args + " pyspark-shell"
        os.environ["PYSPARK_SUBMIT_ARGS"] = submit_args
        spark_conf = init_spark_conf(conf)
        sc = init_internal_nncontext(conf=spark_conf, spark_log_level=self.spark_log_level,
                                     redirect_spark_log=self.redirect_spark_log)
        return sc

    def init_spark_on_local(self, cores, conf=None, python_location=None):
        print("Start to getOrCreate SparkContext")
        if "PYSPARK_PYTHON" not in os.environ:
            os.environ["PYSPARK_PYTHON"] = \
                python_location if python_location else detect_python_location()
        master = "local[{}]".format(cores)
        zoo_conf = init_spark_conf(conf).setMaster(master)
        sc = init_internal_nncontext(conf=zoo_conf, spark_log_level=self.spark_log_level,
                                     redirect_spark_log=self.redirect_spark_log)
        print("Successfully got a SparkContext")
        return sc

    def init_spark_on_yarn(self,
                           hadoop_conf,
                           conda_name,
                           num_executors,
                           executor_cores,
                           executor_memory="2g",
                           driver_cores=4,
                           driver_memory="1g",
                           extra_executor_memory_for_ray=None,
                           extra_python_lib=None,
                           penv_archive=None,
                           additional_archive=None,
                           hadoop_user_name="root",
                           spark_yarn_archive=None,
                           conf=None,
                           jars=None,
                           py_files=None):
        print("Initializing SparkContext for yarn-client mode")
        executor_python_env = "python_env"
        os.environ["HADOOP_CONF_DIR"] = hadoop_conf
        os.environ["HADOOP_USER_NAME"] = hadoop_user_name
        os.environ["PYSPARK_PYTHON"] = "{}/bin/python".format(executor_python_env)

        pack_env = False
        invalidInputError(penv_archive or conda_name,
                          "You should either specify penv_archive or conda_name explicitly")
        try:

            if not penv_archive:
                penv_archive = pack_penv(conda_name, executor_python_env)
                pack_env = True

            archive = "{}#{}".format(penv_archive, executor_python_env)
            if additional_archive:
                archive = archive + "," + additional_archive
            submit_args = "--master yarn --deploy-mode client"
            submit_args = submit_args + " --archives {}".format(archive)
            if py_files:
                submit_args = submit_args + " --py-files {}".format(py_files)
            submit_args = submit_args + gen_submit_args(
                driver_cores, driver_memory, num_executors, executor_cores,
                executor_memory, extra_python_lib, jars)

            conf = enrich_conf_for_spark(conf, driver_cores, driver_memory, num_executors,
                                         executor_cores, executor_memory,
                                         extra_executor_memory_for_ray)
            py_version = ".".join(platform.python_version().split(".")[0:2])
            preload_so = executor_python_env + "/lib/libpython" + py_version + "m.so"
            ld_path = executor_python_env + "/lib:" + executor_python_env + "/lib/python" + \
                py_version + "/lib-dynload"
            if "spark.executor.extraLibraryPath" in conf:
                ld_path = "{}:{}".format(ld_path, conf["spark.executor.extraLibraryPath"])
            conf.update({"spark.scheduler.minRegisteredResourcesRatio": "1.0",
                         "spark.executorEnv.PYTHONHOME": executor_python_env,
                         "spark.executor.extraLibraryPath": ld_path,
                         "spark.executorEnv.LD_PRELOAD": preload_so})
            if spark_yarn_archive:
                conf["spark.yarn.archive"] = spark_yarn_archive
            zoo_bigdl_path_on_executor = ":".join(
                list(get_executor_conda_zoo_classpath(executor_python_env)))
            if "spark.executor.extraClassPath" in conf:
                conf["spark.executor.extraClassPath"] = "{}:{}".format(
                    zoo_bigdl_path_on_executor, conf["spark.executor.extraClassPath"])
            else:
                conf["spark.executor.extraClassPath"] = zoo_bigdl_path_on_executor

            sc = self.create_sc(submit_args, conf)
        finally:
            if conda_name and penv_archive and pack_env:
                os.remove(penv_archive)
        return sc

    def init_spark_on_yarn_cluster(self,
                                   hadoop_conf,
                                   conda_name,
                                   num_executors,
                                   executor_cores,
                                   executor_memory="2g",
                                   driver_cores=4,
                                   driver_memory="1g",
                                   extra_executor_memory_for_ray=None,
                                   extra_python_lib=None,
                                   penv_archive=None,
                                   additional_archive=None,
                                   hadoop_user_name="root",
                                   spark_yarn_archive=None,
                                   conf=None,
                                   jars=None,
                                   py_files=None):
        print("Initializing job for yarn-cluster mode")
        executor_python_env = "python_env"
        os.environ["HADOOP_CONF_DIR"] = hadoop_conf
        os.environ["HADOOP_USER_NAME"] = hadoop_user_name

        pack_env = False
        invalidInputError(penv_archive or conda_name,
                          "You should either specify penv_archive or conda_name explicitly")
        return_value = 1
        try:

            if not penv_archive:
                penv_archive = pack_penv(conda_name, executor_python_env)
                pack_env = True

            archive = "{}#{}".format(penv_archive, executor_python_env)
            if additional_archive:
                archive = archive + "," + additional_archive
            submit_args = "--master yarn --deploy-mode cluster"
            submit_args = submit_args + " --archives {}".format(archive)
            submit_args = submit_args + gen_submit_args(
                driver_cores, driver_memory, num_executors, executor_cores,
                executor_memory, extra_python_lib, jars)
            submit_args = submit_args + " --jars " + ",".join(get_bigdl_jars())
            if py_files:
                submit_args = submit_args + " --py-files {}".format(py_files)

            conf = enrich_conf_for_spark(conf, driver_cores, driver_memory, num_executors,
                                         executor_cores, executor_memory,
                                         extra_executor_memory_for_ray)
            conf["spark.yarn.appMasterEnv.PYSPARK_PYTHON"] = "{}/bin/python".format(
                executor_python_env)
            conf["spark.yarn.appMasterEnv.OnAppMaster"] = "True"
            conf["spark.yarn.appMasterEnv.PYTHONHOME"] = executor_python_env
            conf["spark.executorEnv.PYSPARK_PYTHON"] = "{}/bin/python".format(executor_python_env)
            py_version = ".".join(platform.python_version().split(".")[0:2])
            preload_so = executor_python_env + "/lib/libpython" + py_version + "m.so"
            ld_path = executor_python_env + "/lib:" + executor_python_env + "/lib/python" + \
                py_version + "/lib-dynload"
            if "spark.executor.extraLibraryPath" in conf:
                ld_path = "{}:{}".format(ld_path, conf["spark.executor.extraLibraryPath"])
            conf.update({"spark.scheduler.minRegisteredResourcesRatio": "1.0",
                         "spark.executorEnv.PYTHONHOME": executor_python_env,
                         "spark.executor.extraLibraryPath": ld_path,
                         "spark.executorEnv.LD_PRELOAD": preload_so})
            conf["spark.yarn.appMasterEnv.LD_PRELOAD"] = preload_so
            if spark_yarn_archive:
                conf["spark.yarn.archive"] = spark_yarn_archive
            zoo_bigdl_path_on_executor = ":".join(
                list(get_executor_conda_zoo_classpath(executor_python_env)))
            if "spark.executor.extraClassPath" in conf:
                conf["spark.executor.extraClassPath"] = "{}:{}".format(
                    zoo_bigdl_path_on_executor, conf["spark.executor.extraClassPath"])
            else:
                conf["spark.executor.extraClassPath"] = zoo_bigdl_path_on_executor
            conf["spark.driver.extraClassPath"] = conf["spark.executor.extraClassPath"]

            print(submit_args)
            print(conf)
            conf = " --conf " + " --conf ".join("{}={}".format(*i) for i in conf.items())
            sys_args = " ".join(sys.argv)
            print(sys_args)
            submit_commnad = "spark-submit " + submit_args + " " + conf + " " + sys_args
            print(submit_commnad)
            return_value = os.system(submit_commnad)
        finally:
            if conda_name and penv_archive and pack_env:
                os.remove(penv_archive)
            return return_value

    def init_spark_standalone(self,
                              num_executors,
                              executor_cores,
                              executor_memory="2g",
                              driver_cores=4,
                              driver_memory="1g",
                              master=None,
                              extra_executor_memory_for_ray=None,
                              extra_python_lib=None,
                              conf=None,
                              jars=None,
                              python_location=None,
                              enable_numa_binding=False):
        import subprocess
        import pyspark
        from bigdl.dllib.utils.utils import get_node_ip

        if "PYSPARK_PYTHON" not in os.environ:
            os.environ["PYSPARK_PYTHON"] = \
                python_location if python_location else detect_python_location()
        if not master:
            pyspark_home = os.path.abspath(pyspark.__file__ + "/../")
            zoo_standalone_home = os.path.abspath(__file__ + "/../../../share/dllib/bin/standalone")
            node_ip = get_node_ip()
            SparkRunner.standalone_env = {
                "SPARK_HOME": pyspark_home,
                "ZOO_STANDALONE_HOME": zoo_standalone_home,
                # If not set this, by default master is hostname but not ip,
                "SPARK_MASTER_HOST": node_ip}
            if 'JAVA_HOME' in os.environ:
                SparkRunner.standalone_env["JAVA_HOME"] = os.environ["JAVA_HOME"]
            # The scripts installed from pip don't have execution permission
            # and need to first give them permission.
            pro = subprocess.Popen(["chmod", "-R", "+x", "{}/sbin".format(zoo_standalone_home)])
            os.waitpid(pro.pid, 0)
            # Start master
            start_master_pro = subprocess.Popen(
                "{}/sbin/start-master.sh".format(zoo_standalone_home),
                shell=True, env=SparkRunner.standalone_env)
            _, status = os.waitpid(start_master_pro.pid, 0)
            if status != 0:
                invalidInputError(False, "starting master failed")
            master = "spark://{}:7077".format(node_ip)  # 7077 is the default port
            # Start worker
            if enable_numa_binding:
                worker_script = "start-worker-with-numactl.sh"
                SparkRunner.standalone_env["SPARK_WORKER_INSTANCES"] = str(num_executors)
            else:
                worker_script = "start-worker.sh"
            start_worker_pro = subprocess.Popen(
                "{}/sbin/{} {}".format(zoo_standalone_home, worker_script, master),
                shell=True, env=SparkRunner.standalone_env)
            _, status = os.waitpid(start_worker_pro.pid, 0)
            if status != 0:
                invalidInputError(False, "starting worker failed")
        else:  # A Spark standalone cluster has already been started by the user.
            invalidInputError(master.startswith("spark://"),
                              "Please input a valid master address for your Spark"
                              " standalone cluster: spark://master:port")

        # Start pyspark-shell
        submit_args = "--master " + master
        submit_args = submit_args + gen_submit_args(
            driver_cores, driver_memory, num_executors, executor_cores,
            executor_memory, extra_python_lib, jars)

        conf = enrich_conf_for_spark(conf, driver_cores, driver_memory, num_executors,
                                     executor_cores, executor_memory, extra_executor_memory_for_ray)
        conf.update({
            "spark.cores.max": num_executors * executor_cores,
            "spark.executorEnv.PYTHONHOME": "/".join(detect_python_location().split("/")[:-2])
        })
        # Driver and executor are assumed to have the same Python environment
        zoo_bigdl_jar_path = get_zoo_bigdl_classpath_on_driver()
        if "spark.executor.extraClassPath" in conf:
            conf["spark.executor.extraClassPath"] = "{}:{}".format(
                zoo_bigdl_jar_path, conf["spark.executor.extraClassPath"])
        else:
            conf["spark.executor.extraClassPath"] = zoo_bigdl_jar_path

        sc = self.create_sc(submit_args, conf)
        return sc

    @staticmethod
    def stop_spark_standalone():
        import subprocess
        env = SparkRunner.standalone_env
        if env is not None:
            stop_worker_pro = subprocess.Popen(
                "{}/sbin/stop-worker.sh".format(env["ZOO_STANDALONE_HOME"]), shell=True, env=env)
            os.waitpid(stop_worker_pro.pid, 0)
            stop_master_pro = subprocess.Popen(
                "{}/sbin/stop-master.sh".format(env["ZOO_STANDALONE_HOME"]), shell=True, env=env)
            os.waitpid(stop_master_pro.pid, 0)
        else:
            # if env is None, then the standalone cluster is not started by analytics zoo
            pass

    def init_spark_on_k8s(self,
                          master,
                          container_image,
                          conda_name,
                          num_executors,
                          executor_cores,
                          executor_memory="2g",
                          driver_memory="1g",
                          driver_cores=4,
                          extra_executor_memory_for_ray=None,
                          extra_python_lib=None,
                          penv_archive=None,
                          conf=None,
                          jars=None,
                          python_location=None):
        print("Initializing SparkContext for k8s-client mode")
        executor_python_env = "python_env"
        os.environ["PYSPARK_PYTHON"] = "{}/bin/python".format(executor_python_env)
        pack_env = False
        invalidInputError(penv_archive or conda_name,
                          "You should either specify penv_archive or conda_name explicitly")
        try:

            if not penv_archive:
                penv_archive = pack_penv(conda_name, executor_python_env)
                pack_env = True

            archive = "{}#{}".format(penv_archive, executor_python_env)

            submit_args = "--master " + master + " --deploy-mode client"
            submit_args = submit_args + " --archives {}".format(archive)
            submit_args = submit_args + gen_submit_args(
                driver_cores, driver_memory, num_executors, executor_cores,
                executor_memory, extra_python_lib, jars)
            submit_args = submit_args + " --jars " + ",".join(get_bigdl_jars())

            conf = enrich_conf_for_spark(conf, driver_cores, driver_memory, num_executors,
                                         executor_cores, executor_memory,
                                         extra_executor_memory_for_ray)
            py_version = ".".join(platform.python_version().split(".")[0:2])
            preload_so = executor_python_env + "/lib/libpython" + py_version + "m.so"
            ld_path = executor_python_env + "/lib:" + executor_python_env + "/lib/python" + \
                py_version + "/lib-dynload"
            image_workdir = get_bigdl_image_workdir()
            tf_libs_path = image_workdir + "/" + executor_python_env + "/lib/python" + \
                py_version + "/site-packages/bigdl/share/tflibs"
            if "spark.executor.extraLibraryPath" in conf:
                ld_path = "{}:{}".format(ld_path, conf["spark.executor.extraLibraryPath"])
            conf.update({"spark.cores.max": num_executors * executor_cores,
                         "spark.executorEnv.PYTHONHOME": executor_python_env,
                         "spark.executorEnv.TF_LIBS_PATH": tf_libs_path,
                         "spark.executor.extraLibraryPath": ld_path,
                         "spark.executorEnv.LD_PRELOAD": preload_so,
                         "spark.kubernetes.container.image": container_image})
            if "spark.driver.host" not in conf:
                conf["spark.driver.host"] = get_node_ip()
            if "spark.driver.port" not in conf:
                conf["spark.driver.port"] = random.randint(10000, 65535)

            sc = self.create_sc(submit_args, conf)
        finally:
            if conda_name and penv_archive and pack_env:
                os.remove(penv_archive)
        return sc

    def init_spark_on_k8s_cluster(self,
                                  master,
                                  container_image,
                                  num_executors,
                                  executor_cores,
                                  executor_memory="2g",
                                  driver_memory="1g",
                                  driver_cores=4,
                                  extra_executor_memory_for_ray=None,
                                  extra_python_lib=None,
                                  penv_archive=None,
                                  conf=None,
                                  jars=None,
                                  python_location=None):
        print("Initializing SparkContext for k8s-cluster mode")
        executor_python_env = "python_env"

        invalidInputError(penv_archive,
                          "You should specify penv_archive explicitly for k8s cluster mode")

        archive = "{}#{}".format(penv_archive, executor_python_env)

        submit_args = "--master " + master + " --deploy-mode cluster"
        submit_args = submit_args + " --archives {}".format(archive)
        submit_args = submit_args + gen_submit_args(
            driver_cores, driver_memory, num_executors, executor_cores,
            executor_memory, extra_python_lib, jars)

        conf = enrich_conf_for_spark(conf, driver_cores, driver_memory, num_executors,
                                     executor_cores, executor_memory, extra_executor_memory_for_ray)

        conf["spark.pyspark.driver.python"] = "{}/bin/python".format(executor_python_env)
        conf["spark.pyspark.python"] = "{}/bin/python".format(executor_python_env)
        conf["spark.kubernetes.driverEnv.onDriver"] = "True"

        py_version = ".".join(platform.python_version().split(".")[0:2])
        preload_so = executor_python_env + "/lib/libpython" + py_version + "m.so"
        ld_path = executor_python_env + "/lib:" + executor_python_env + "/lib/python" + \
            py_version + "/lib-dynload"
        if "spark.executor.extraLibraryPath" in conf:
            ld_path = "{}:{}".format(ld_path, conf["spark.executor.extraLibraryPath"])
        conf.update({"spark.cores.max": num_executors * executor_cores,
                     "spark.executorEnv.PYTHONHOME": executor_python_env,
                     "spark.executor.extraLibraryPath": ld_path,
                     "spark.executorEnv.LD_PRELOAD": preload_so,
                     "spark.kubernetes.container.image": container_image})
        if "spark.executor.extraClassPath" in conf:
            conf["spark.executor.extraClassPath"] = "{}:{}".format(
                zoo_bigdl_path_on_executor, conf["spark.executor.extraClassPath"])
        else:
            # BigDL Class path in k8s image
            bigdl_class_version = get_bigdl_class_version()
            conf["spark.executor.extraClassPath"] = "/opt/" + bigdl_class_version + "bigdl-" \
                + "/jars/*"
        conf["spark.driver.extraClassPath"] = conf["spark.executor.extraClassPath"]
        sys_args = "local://" + " ".join(sys.argv)
        conf = " --conf " + " --conf ".join("{}={}".format(*i) for i in conf.items())
        submit_commnad = "spark-submit " + submit_args + " " + conf + " " + sys_args
        print("submit command", submit_commnad)
        return_value = os.system(submit_commnad)
        return return_value


def gen_submit_args(driver_cores, driver_memory, num_executors, executor_cores, executor_memory,
                    extra_python_lib=None, jars=None):
    submit_args = " --driver-cores {} --driver-memory {} --num-executors {}" \
                  " --executor-cores {} --executor-memory {}" \
        .format(driver_cores, driver_memory, num_executors, executor_cores, executor_memory)
    if extra_python_lib:
        submit_args = submit_args + " --py-files {}".format(extra_python_lib)
    if jars:
        submit_args = submit_args + " --jars {}".format(jars)
    return submit_args


def enrich_conf_for_spark(conf, driver_cores, driver_memory, num_executors, executor_cores,
                          executor_memory, extra_executor_memory_for_ray=None):
    if not conf:
        conf = {}
    conf.update({"spark.driver.cores": driver_cores,
                 "spark.driver.memory": driver_memory,
                 "spark.executor.instances": num_executors,
                 "spark.executor.cores": executor_cores,
                 "spark.executor.memory": executor_memory})
    if extra_executor_memory_for_ray:
        conf["spark.executor.memoryOverhead"] = extra_executor_memory_for_ray
    return conf
