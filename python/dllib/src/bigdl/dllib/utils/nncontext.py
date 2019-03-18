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

from bigdl.util.common import *
import warnings


def init_nncontext(conf=None):
    """
    Creates or gets a SparkContext with optimized configuration for BigDL performance.
    The method will also initialize the BigDL engine.

    Note: if you use spark-shell or Jupyter notebook, as the Spark context is created
    before your code, you have to set Spark conf values through command line options
    or properties file, and init BigDL engine manually.

    :param conf: User defined Spark conf
    """
    if isinstance(conf, six.string_types):
        sc = getOrCreateSparkContext(conf=None, appName=conf)
    else:
        sc = getOrCreateSparkContext(conf=conf)
    check_version()
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()
    return sc


def getOrCreateSparkContext(conf=None, appName=None):
    """
    Get the current active spark context and create one if no active instance
    :param conf: combining bigdl configs into spark conf
    :return: SparkContext
    """

    with SparkContext._lock:
        if SparkContext._active_spark_context is None:
            spark_conf = init_spark_conf() if conf is None else conf
            if appName:
                spark_conf.setAppName(appName)
            return SparkContext.getOrCreate(spark_conf)
        else:
            return SparkContext.getOrCreate()


def get_analytics_zoo_conf():
    zoo_conf_file = "spark-analytics-zoo.conf"
    zoo_python_wrapper = "python-api.zip"

    for p in sys.path:
        if zoo_conf_file in p and os.path.isfile(p):
            with open(p) if sys.version_info < (3,) else open(p, encoding='latin-1') as conf_file:
                return load_conf(conf_file.read())
        if zoo_python_wrapper in p and os.path.isfile(p):
            import zipfile
            with zipfile.ZipFile(p, 'r') as zip_conf:
                if zoo_conf_file in zip_conf.namelist():
                    content = zip_conf.read(zoo_conf_file)
                    if sys.version_info >= (3,):
                        content = str(content, 'latin-1')
                    return load_conf(content)
    return {}


def init_spark_conf():
    zoo_conf = get_analytics_zoo_conf()
    sparkConf = SparkConf()
    sparkConf.setAll(zoo_conf.items())
    if os.environ.get("BIGDL_JARS", None) and not is_spark_below_2_2():
        for jar in os.environ["BIGDL_JARS"].split(":"):
            extend_spark_driver_cp(sparkConf, jar)

    # add content in PYSPARK_FILES in spark.submit.pyFiles
    # This is a workaround for current Spark on k8s
    python_lib = os.environ.get('PYSPARK_FILES', None)
    if python_lib:
        existing_py_files = sparkConf.get("spark.submit.pyFiles")
        if existing_py_files:
            sparkConf.set(key="spark.submit.pyFiles",
                          value="%s,%s" % (python_lib, existing_py_files))
        else:
            sparkConf.set(key="spark.submit.pyFiles", value=python_lib)

    return sparkConf


def check_version():
    sc = getOrCreateSparkContext()
    conf = sc._conf
    if conf.get("spark.analytics.zoo.versionCheck", "False").lower() == "true":
        report_warn = conf.get(
            "spark.analytics.zoo.versionCheck.warning", "False").lower() == "true"
        _check_spark_version(sc, report_warn)


def _split_full_version(version):
    parts = version.split(".")
    major = parts[0]
    feature = parts[1]
    maintenance = parts[2]
    return (major, feature, maintenance)


def _check_spark_version(sc, report_warn):
    version_info = _get_bigdl_verion_conf()
    (c_major, c_feature, c_maintenance) = _split_full_version(version_info['spark_version'])
    (r_major, r_feature, r_maintenance) = _split_full_version(sc.version)
    error_message = \
        """
        The compile time spark version is not compatible with the spark runtime version.
        Compile time version is %s, runtime version is %s. If you want bypass this check,
        please set spark.analytics.zoo.versionCheck to false, and if you want to only report
        an warning message, please set spark.analytics.zoo.versionCheck.warning to true.
        """ % (version_info['spark_version'], sc.version)
    if c_major != r_major:
        if not report_warn:
            print("***************************Usage Error*****************************")
            print(error_message)
            raise RuntimeError(error_message)
        else:
            warnings.warn(error_message)
    elif not (c_maintenance == r_maintenance and c_feature == r_feature):
        warnings.warn("The compile time spark version may not compatible with " +
                      "the Spark runtime version. " +
                      "Compile time version is %s, " % version_info['spark_version'] +
                      "runtime version is %s" % sc.version)


def _get_bigdl_verion_conf():
    bigdl_build_file = "zoo-version-info.properties"
    bigdl_python_wrapper = "python-api.zip"

    for p in sys.path:
        if bigdl_build_file in p and os.path.isfile(p):
            with open(p) if sys.version_info < (3,) else open(p, encoding='latin-1') as conf_file:
                return load_conf(conf_file.read(), "=")
        if bigdl_python_wrapper in p and os.path.isfile(p):
            import zipfile
            with zipfile.ZipFile(p, 'r') as zip_conf:
                if bigdl_build_file in zip_conf.namelist():
                    content = zip_conf.read(bigdl_build_file)
                    if sys.version_info >= (3,):
                        content = str(content, 'latin-1')
                    return load_conf(content, "=")
    raise RuntimeError("Error while locating file zoo-version-info.properties, " +
                       "please make sure the mvn generate-resources phase" +
                       " is executed and a zoo-version-info.properties file" +
                       " is located in zoo/target/extra-resources")


def load_conf(conf_str, split_char=None):
    return dict(line.split(split_char) for line in conf_str.split("\n") if
                "#" not in line and line.strip())
