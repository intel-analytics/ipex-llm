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
        py4j = glob.glob(os.path.join(spark_home, 'python/lib', 'py4j-*.zip'))[0]
        pyspark = glob.glob(os.path.join(spark_home, 'python/lib', 'pyspark*.zip'))[0]
        sys.path.insert(0, py4j)
        sys.path.insert(0, pyspark)


def __prepare_bigdl_env():
    import bigdl.nn.layer
    jar_dir = os.path.abspath(bigdl.nn.layer.__file__ + "/../../")
    jar_paths = glob.glob(os.path.join(jar_dir, "share/lib/*.jar"))
    conf_paths = glob.glob(os.path.join(jar_dir, "share/conf/*.conf"))

    def append_path(env_var_name, path):
        try:
            os.environ[env_var_name] = path + ":" + os.environ[
                env_var_name]  # noqa
        except KeyError:
            os.environ[env_var_name] = path

    if conf_paths and conf_paths:
        assert len(conf_paths) == 1, "Expecting one jar: %s" % len(jar_paths)
        assert len(conf_paths) == 1, "Expecting one conf: %s" % len(conf_paths)
        print("Adding %s to spark.driver.extraClassPath" % jar_paths[0])
        print("Adding %s to spark.executor.extraClassPath" % jar_paths[0])
        append_path("spark.driver.extraClassPath", jar_paths[0])
        append_path("spark.executor.extraClassPath", jar_paths[0])
        append_path("SPARK_CLASSPATH", jar_paths[0])
        print("Prepending %s to sys.path" % conf_paths[0])
        sys.path.insert(0, conf_paths[0])


def prepare_env():
    __prepare_spark_env()
    __prepare_bigdl_env()