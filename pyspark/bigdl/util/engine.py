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

import sys
import os
import glob

def __prepare_spark_env():
    modules = sys.modules
    if "pyspark" not in modules or "py4j" not in modules:
        spark_home = os.environ.get('SPARK_HOME', None)
        if not spark_home:
            raise ValueError(
                """Could not find Spark. Pls make sure SPARK_HOME env is set:
                   export SPARK_HOME=path to your spark home directory""")
        print("Using %s" % spark_home)
        py4j = glob.glob(os.path.join(spark_home, 'python/lib', 'py4j-*.zip'))[0]
        pyspark = glob.glob(os.path.join(spark_home, 'python/lib', 'pyspark*.zip'))[0]
        sys.path.insert(0, py4j)
        sys.path.insert(0, pyspark)


def __prepare_bigdl_env():
    jar_dir = os.path.abspath(__file__ + "/../../")
    jar_paths = glob.glob(os.path.join(jar_dir, "share/lib/*.jar"))
    conf_paths = glob.glob(os.path.join(jar_dir, "share/conf/*.conf"))

    def append_path(env_var_name, path):
        try:
            print("Adding %s to %s" % (jar_paths[0], env_var_name))
            os.environ[env_var_name] = path + ":" + os.environ[
                env_var_name]  # noqa
        except KeyError:
            os.environ[env_var_name] = path

    if conf_paths and conf_paths:
        assert len(conf_paths) == 1, "Expecting one jar: %s" % len(jar_paths)
        assert len(conf_paths) == 1, "Expecting one conf: %s" % len(conf_paths)
        append_path("SPARK_CLASSPATH", jar_paths[0])
        print("Prepending %s to sys.path" % conf_paths[0])
        sys.path.insert(0, conf_paths[0])
    # Get rid of source bigdl.sh and These have been remove in master branch
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['KMP_BLOCKTIME'] = '0'
    os.environ['OMP_WAIT_POLICY'] = 'passive'
    os.environ['DL_ENGINE_TYPE'] = 'mklblas'
    os.environ['MKL_DISABLE_FAST_MM'] = '1'


def prepare_env():
    __prepare_spark_env()
    __prepare_bigdl_env()