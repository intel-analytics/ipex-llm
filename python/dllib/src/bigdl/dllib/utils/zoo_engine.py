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

import sys
import os
import glob
import warnings


def exist_pyspark():
    # check whether pyspark package exists
    try:
        import pyspark
        return True
    except ImportError:
        return False


def check_spark_source_conflict(spark_home, pyspark_path):
    # Check if both $SPARK_HOME and pyspark package exist.
    # Trigger a warning if two spark sources don't match.
    if spark_home and not pyspark_path.startswith(spark_home):
        warning_msg = "Find both SPARK_HOME and pyspark. You may need to check whether they " + \
                      "match with each other. SPARK_HOME environment variable " + \
                      "is set to: " + spark_home + \
                      ", and pyspark is found in: " + pyspark_path + ". If they are unmatched, " + \
                      "you are recommended to use one source only to avoid conflict. " + \
                      "For example, you can unset SPARK_HOME and use pyspark only."
        warnings.warn(warning_msg)


def __prepare_spark_env():
    spark_home = os.environ.get('SPARK_HOME', None)
    if exist_pyspark():
        # use pyspark as the spark source
        import pyspark
        check_spark_source_conflict(spark_home, pyspark.__file__)
    else:
        # use $SPARK_HOME as the spark source
        if not spark_home:
            raise ValueError(
                """Could not find Spark. Please make sure SPARK_HOME env is set:
                   export SPARK_HOME=path to your spark home directory.""")
        print("Using %s" % spark_home)
        py4j = glob.glob(os.path.join(spark_home, 'python/lib', 'py4j-*.zip'))[0]
        pyspark = glob.glob(os.path.join(spark_home, 'python/lib', 'pyspark*.zip'))[0]
        if py4j not in sys.path:
            sys.path.insert(0, py4j)
        if pyspark not in sys.path:
            sys.path.insert(0, pyspark)


def __prepare_analytics_zoo_env():
    jar_dir = os.path.abspath(__file__ + "/../../")
    conf_paths = glob.glob(os.path.join(jar_dir, "share/conf/*.conf"))
    extra_resources_paths = glob.glob(os.path.join(jar_dir, "share/extra-resources/*"))
    analytics_zoo_classpath = get_analytics_zoo_classpath()

    def append_path(env_var_name, path):
        try:
            if path not in os.environ[env_var_name]:
                print("Adding %s to %s" % (path, env_var_name))
                os.environ[env_var_name] = path + ":" + os.environ[env_var_name]  # noqa
        except KeyError:
            os.environ[env_var_name] = path

    if analytics_zoo_classpath:
        append_path("BIGDL_JARS", analytics_zoo_classpath)

    if conf_paths:
        assert len(conf_paths) == 1, "Expecting one conf, but got: %s" % len(conf_paths)
        if conf_paths[0] not in sys.path:
            print("Prepending %s to sys.path" % conf_paths[0])
            sys.path.insert(0, conf_paths[0])

    if extra_resources_paths:
        for resource in extra_resources_paths:
            if resource not in extra_resources_paths:
                print("Prepending %s to sys.path" % resource)
                sys.path.insert(0, resource)

    if os.environ.get("BIGDL_JARS", None) and is_spark_below_2_2():
        for jar in os.environ["BIGDL_JARS"].split(":"):
            append_path("SPARK_CLASSPATH", jar)

    if os.environ.get("BIGDL_PACKAGES", None):
        for package in os.environ["BIGDL_PACKAGES"].split(":"):
            if package not in sys.path:
                sys.path.insert(0, package)

    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def get_analytics_zoo_classpath():
    """
    Get and return the jar path for analytics-zoo if exists.
    """
    if os.getenv("BIGDL_CLASSPATH"):
        return os.environ["BIGDL_CLASSPATH"]
    jar_dir = os.path.abspath(__file__ + "/../../")
    jar_paths = glob.glob(os.path.join(jar_dir, "share/lib/*.jar"))
    if jar_paths:
        assert len(jar_paths) == 1, "Expecting one jar: %s" % len(jar_paths)
        return jar_paths[0]
    return ""


def is_spark_below_2_2():
    """
    Check if spark version is below 2.2.
    """
    import pyspark
    if hasattr(pyspark, "version"):
        full_version = pyspark.version.__version__
        # We only need the general spark version (eg, 1.6, 2.2).
        parts = full_version.split(".")
        spark_version = parts[0] + "." + parts[1]
        if compare_version(spark_version, "2.2") >= 0:
            return False
    return True


def compare_version(version1, version2):
    """
    Compare version strings.
    Return 1 if version1 is after version2;
          -1 if version1 is before version2;
           0 if two versions are the same.
    """
    v1_arr = version1.split(".")
    v2_arr = version2.split(".")
    len1 = len(v1_arr)
    len2 = len(v2_arr)
    len_max = max(len1, len2)
    for x in range(len_max):
        v1_token = 0
        if x < len1:
            v1_token = int(v1_arr[x])
        v2_token = 0
        if x < len2:
            v2_token = int(v2_arr[x])
        if v1_token < v2_token:
            return -1
        if v1_token > v2_token:
            return 1
    return 0


def prepare_env():
    __prepare_spark_env()
    __prepare_analytics_zoo_env()
