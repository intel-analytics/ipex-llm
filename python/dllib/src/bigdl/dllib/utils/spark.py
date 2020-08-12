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
import glob

from pyspark import SparkContext
from zoo.common.nncontext import init_spark_conf
from zoo import init_nncontext


class SparkRunner:
    standalone_env = None

    def __init__(self,
                 spark_log_level="WARN",
                 redirect_spark_log=True):
        self.spark_log_level = spark_log_level
        self.redirect_spark_log = redirect_spark_log
        self.PYTHON_ENV = "python_env"
        with SparkContext._lock:
            if SparkContext._active_spark_context:
                raise Exception("There's existing SparkContext. Please close it first.")
        import pyspark
        print("Current pyspark location is : {}".format(pyspark.__file__))

    # This is adopted from conda-pack.
    def _pack_conda_main(self, args):
        import sys
        import traceback
        from conda_pack.cli import fail, PARSER, context
        import conda_pack
        from conda_pack import pack, CondaPackException
        args = PARSER.parse_args(args=args)
        # Manually handle version printing to output to stdout in python < 3.4
        if args.version:
            print('conda-pack %s' % conda_pack.__version__)
            sys.exit(0)

        try:
            with context.set_cli():
                pack(name=args.name,
                     prefix=args.prefix,
                     output=args.output,
                     format=args.format,
                     force=args.force,
                     compress_level=args.compress_level,
                     n_threads=args.n_threads,
                     zip_symlinks=args.zip_symlinks,
                     zip_64=not args.no_zip_64,
                     arcroot=args.arcroot,
                     dest_prefix=args.dest_prefix,
                     verbose=not args.quiet,
                     filters=args.filters)
        except CondaPackException as e:
            fail("CondaPackError: %s" % e)
        except KeyboardInterrupt:
            fail("Interrupted")
        except Exception:
            fail(traceback.format_exc())

    def pack_penv(self, conda_name):
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        tmp_path = "{}/{}.tar.gz".format(tmp_dir, self.PYTHON_ENV)
        print("Start to pack current python env")
        self._pack_conda_main(["--output", tmp_path, "--n-threads", "8", "--name", conda_name])
        print("Packing has been completed: {}".format(tmp_path))
        return tmp_path

    def _create_sc(self, submit_args, conf):
        os.environ['PYSPARK_SUBMIT_ARGS'] = submit_args
        zoo_conf = init_spark_conf(conf)
        sc = init_nncontext(conf=zoo_conf, redirect_spark_log=self.redirect_spark_log)
        sc.setLogLevel(self.spark_log_level)

        return sc

    def _detect_python_location(self):
        import subprocess
        pro = subprocess.Popen(
            "command -v python",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        out, err = pro.communicate()
        out = out.decode("utf-8")
        err = err.decode("utf-8")
        errorcode = pro.returncode
        if 0 != errorcode:
            raise Exception(err +
                            "Cannot detect current python location."
                            "Please set it manually by python_location")
        return out.strip()

    def _get_conda_python_path(self):
        conda_env_path = "/".join(self._detect_python_location().split("/")[:-2])
        python_interpreters = glob.glob("{}/lib/python*".format(conda_env_path))
        assert len(python_interpreters) == 1, \
            "Conda env should contain a single python, but got: {}".format(python_interpreters)
        return python_interpreters[0]

    def _get_conda_python_intepreter_name(self):
        return self._get_conda_python_path().split("/")[-1]

    def _get_bigdl_classpath_jar_name_on_driver(self):
        from bigdl.util.engine import get_bigdl_classpath
        bigdl_classpath = get_bigdl_classpath()
        assert bigdl_classpath, "Cannot find bigdl classpath"
        return bigdl_classpath.split("/")[-1]

    def _get_zoo_classpath_jar_name_on_driver(self):
        from zoo.util.engine import get_analytics_zoo_classpath
        zoo_classpath = get_analytics_zoo_classpath()
        assert zoo_classpath, "Cannot find Analytics-Zoo classpath"
        return zoo_classpath.split("/")[-1]

    def _assemble_zoo_classpath_for_executor(self, conda_path=None):
        if not conda_path:
            conda_path = self.PYTHON_ENV
        prefix = "{}/lib/{}/site-packages/"\
            .format(conda_path, self._get_conda_python_intepreter_name())
        return ["{}/zoo/share/lib/{}".format(
                prefix, self._get_zoo_classpath_jar_name_on_driver()),
                "{}/bigdl/share/lib/{}".format(
                prefix, self._get_bigdl_classpath_jar_name_on_driver())]

    def init_spark_on_local(self, cores, conf=None, python_location=None):
        print("Start to getOrCreate SparkContext")
        if 'PYSPARK_PYTHON' not in os.environ:
            os.environ['PYSPARK_PYTHON'] = \
                python_location if python_location else self._detect_python_location()
        master = "local[{}]".format(cores)
        zoo_conf = init_spark_conf(conf).setMaster(master)
        sc = init_nncontext(conf=zoo_conf, redirect_spark_log=self.redirect_spark_log)
        sc.setLogLevel(self.spark_log_level)
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
                           jars=None):
        os.environ["HADOOP_CONF_DIR"] = hadoop_conf
        os.environ['HADOOP_USER_NAME'] = hadoop_user_name
        os.environ['PYSPARK_PYTHON'] = "{}/bin/python".format(self.PYTHON_ENV)

        def _yarn_opt(jars):

            archive = "{}#{}".format(penv_archive, self.PYTHON_ENV)
            if additional_archive:
                archive = archive + "," + additional_archive
            command = " --archives {} --driver-cores {} --driver-memory {}" \
                      " --num-executors {} --executor-cores {} --executor-memory {}". \
                format(archive, driver_cores, driver_memory,
                       num_executors, executor_cores, executor_memory)

            if extra_python_lib:
                command = command + " --py-files {}".format(extra_python_lib)
            if jars:
                command = command + " --jars {}".format(jars)
            return command

        def _submit_opt():
            conf = {
                "spark.driver.memory": driver_memory,
                "spark.driver.cores": driver_cores,
                "spark.executor.cores": executor_cores,
                "spark.executor.memory": executor_memory,
                "spark.scheduler.minRegisteredResourcesRatio": "1.0"}
            if extra_executor_memory_for_ray:
                conf["spark.executor.memoryOverhead"] = extra_executor_memory_for_ray
            if spark_yarn_archive:
                conf["spark.yarn.archive"] = spark_yarn_archive
            return " --master yarn --deploy-mode client" + _yarn_opt(jars) + ' pyspark-shell ', conf

        pack_env = False
        assert penv_archive or conda_name, \
            "You should either specify penv_archive or conda_name explicitly"
        try:
            if not penv_archive:
                penv_archive = self.pack_penv(conda_name)
                pack_env = True

            submit_args, _conf = _submit_opt()

            if not conf:
                conf = {}
            zoo_bigdl_path_on_executor = ":".join(self._assemble_zoo_classpath_for_executor())

            if "spark.executor.extraClassPath" in conf:
                conf["spark.executor.extraClassPath"] = "{}:{}".format(
                    zoo_bigdl_path_on_executor, conf["spark.executor.extraClassPath"])
            else:
                conf["spark.executor.extraClassPath"] = zoo_bigdl_path_on_executor

            conf["spark.executorEnv.PYTHONHOME"] = self.PYTHON_ENV

            for item in conf.items():
                _conf[str(item[0])] = str(item[1])
            sc = self._create_sc(submit_args, _conf)
        finally:
            if conda_name and penv_archive and pack_env:
                os.remove(penv_archive)
        return sc

    def init_spark_standalone(self,
                              num_executors,
                              executor_cores,
                              executor_memory="10g",
                              driver_memory="1g",
                              driver_cores=4,
                              spark_master=None,
                              extra_executor_memory_for_ray=None,
                              extra_python_lib=None,
                              conf=None,
                              jars=None):
        import subprocess
        import pyspark
        from zoo.util.utils import get_node_ip
        from zoo.util.engine import get_analytics_zoo_classpath
        from bigdl.util.engine import get_bigdl_classpath

        if 'PYSPARK_PYTHON' not in os.environ:
            os.environ["PYSPARK_PYTHON"] = self._detect_python_location()
        if not spark_master:
            pyspark_home = os.path.abspath(pyspark.__file__ + "/../")
            zoo_standalone_home = os.path.abspath(__file__ + "/../../share/bin/standalone")
            node_ip = get_node_ip()
            SparkRunner.standalone_env = {
                "SPARK_HOME": pyspark_home,
                "ZOO_STANDALONE_HOME": zoo_standalone_home,
                # If not set this, by default master is hostname but not ip,
                "SPARK_MASTER_HOST": node_ip}
            # The scripts installed from pip don't have execution permission
            # and need to first give them permission.
            pro = subprocess.Popen(["chmod", "-R", "+x", "{}/sbin".format(zoo_standalone_home)])
            os.waitpid(pro.pid, 0)
            # Start master
            start_master_pro = subprocess.Popen(
                "{}/sbin/start-master.sh".format(zoo_standalone_home),
                shell=True, env=SparkRunner.standalone_env)
            os.waitpid(start_master_pro.pid, 0)
            spark_master = "spark://{}:7077".format(node_ip)  # 7077 is the default port
            # Start worker
            start_worker_pro = subprocess.Popen(
                "{}/sbin/start-worker.sh {}".format(zoo_standalone_home, spark_master),
                shell=True, env=SparkRunner.standalone_env)
            os.waitpid(start_worker_pro.pid, 0)
        else:  # A Spark standalone cluster has already been started by the user.
            assert spark_master.startswith("spark://"), \
                "Please input a valid master address for Spark standalone: spark://master:port"

        # Start pyspark-shell
        submit_args = " --master " + spark_master
        submit_args = submit_args + " --driver-cores {} --driver-memory {} --num-executors {}" \
                                    " --executor-cores {} --executor-memory {}"\
            .format(driver_cores, driver_memory, num_executors, executor_cores, executor_memory)
        if extra_python_lib:
            submit_args = submit_args + " --py-files {}".format(extra_python_lib)
        if jars:
            submit_args = submit_args + " --jars {}".format(jars)
        submit_args = submit_args + " pyspark-shell"
        os.environ['PYSPARK_SUBMIT_ARGS'] = submit_args

        zoo_bigdl_jar_path = ":".join([get_analytics_zoo_classpath(), get_bigdl_classpath()])
        spark_conf = init_spark_conf(conf) \
            .set("spark.driver.cores", driver_cores) \
            .set("spark.driver.memory", driver_memory) \
            .set("spark.executor.instances", num_executors) \
            .set("spark.executor.cores", executor_cores) \
            .set("spark.cores.max", num_executors * executor_cores) \
            .set("spark.executorEnv.PYTHONHOME",
                 "/".join(self._detect_python_location().split("/")[:-2]))
        if extra_executor_memory_for_ray:
            spark_conf.set("spark.executor.memoryOverhead",
                           extra_executor_memory_for_ray)
        if spark_conf.contains("spark.executor.extraClassPath"):
            spark_conf.set("spark.executor.extraClassPath", "{}:{}".format(
                zoo_bigdl_jar_path, conf.get("spark.executor.extraClassPath")))
        else:
            spark_conf.set("spark.executor.extraClassPath", zoo_bigdl_jar_path)

        sc = init_nncontext(spark_conf, redirect_spark_log=self.redirect_spark_log)
        sc.setLogLevel(self.spark_log_level)
        return sc

    @staticmethod
    def stop_spark_standalone():
        import subprocess
        env = SparkRunner.standalone_env
        if not env:
            import pyspark
            pyspark_home = os.path.abspath(pyspark.__file__ + "/../")
            zoo_standalone_home = os.path.abspath(__file__ + "/../../share/bin/standalone")
            env = {"SPARK_HOME": pyspark_home,
                   "ZOO_STANDALONE_HOME": zoo_standalone_home}
        stop_worker_pro = subprocess.Popen(
            "{}/sbin/stop-worker.sh".format(env["ZOO_STANDALONE_HOME"]), shell=True, env=env)
        os.waitpid(stop_worker_pro.pid, 0)
        stop_master_pro = subprocess.Popen(
            "{}/sbin/stop-master.sh".format(env["ZOO_STANDALONE_HOME"]), shell=True, env=env)
        os.waitpid(stop_master_pro.pid, 0)
