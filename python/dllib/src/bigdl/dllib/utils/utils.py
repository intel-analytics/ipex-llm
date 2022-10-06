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

import glob
from bigdl.dllib.utils.file_utils import Sample
import numpy as np
from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
import sys
import os
import re
from bigdl.dllib.utils.log4Error import *


if sys.version >= '3':
    long = int
    unicode = str


def to_sample_rdd(x, y, sc, num_slices=None):
    """
    Convert x and y into RDD[Sample]
    :param sc: SparkContext
    :param x: ndarray and the first dimension should be batch
    :param y: ndarray and the first dimension should be batch
    :param num_slices: The number of partitions for x and y.
    :return:
    """
    x_rdd = sc.parallelize(x, num_slices)
    y_rdd = sc.parallelize(y, num_slices)
    return x_rdd.zip(y_rdd).map(lambda item: Sample.from_ndarray(item[0], item[1]))


def get_node_ip():
    """
    This function is ported from ray to get the ip of the current node. In the settings where
    Ray is not involved, calling ray._private.services.get_node_ip_address would introduce
    Ray overhead.
    """
    import socket
    import errno
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This command will throw an exception if there is no internet connection.
        s.connect(("8.8.8.8", 80))
        node_ip_address = s.getsockname()[0]
    except OSError as e:
        node_ip_address = "127.0.0.1"
        # [Errno 101] Network is unreachable
        if e.errno == errno.ENETUNREACH:
            try:
                # try get node ip address from host name
                host_name = socket.getfqdn(socket.gethostname())
                node_ip_address = socket.gethostbyname(host_name)
            except Exception:
                pass
    finally:
        s.close()
    return node_ip_address


def detect_python_location():
    import sys
    return sys.executable


def detect_conda_env_name():
    # if call on yarn app master, return empty string
    if os.environ.get("OnAppMaster", "False") == "True":
        return ""
    # This only works for anaconda3
    import subprocess
    pro = subprocess.Popen(
        "conda info",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = pro.communicate()
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    errorcode = pro.returncode
    if 0 != errorcode:
        invalidInputError(False, err +
                          "Cannot find conda info. Please verify your conda installation")
    for line in out.split('\n'):
        item = line.split(':')
        if len(item) == 2:
            if item[0].strip() == "active environment":
                return item[1].strip()
    # For anaconda2 or if any error occurs above
    python_location = detect_python_location()
    if "envs" in python_location:
        return python_location.split("/")[-3]
        invalidInputError(False,
                          err + "Failed to detect the current conda environment. Please verify "
                                "your conda installation and activate the env you want to use")


# This is adopted from conda-pack.
def pack_conda_main(conda_name, tmp_path):
    import subprocess
    pack_env = os.environ.copy()
    if "PYTHONHOME" in pack_env:
        pack_env.pop("PYTHONHOME")
    pack_cmd = "conda pack --format tar.gz --n-threads 8 -f -n {} -o {}" \
        .format(conda_name, tmp_path)
    pro = subprocess.Popen(pack_cmd, shell=True, env=pack_env)
    if pro.wait() != 0:
        invalidInputError(False, f"conda pack failed! Error executing command: {pack_cmd} ")


def pack_penv(conda_name, output_name):
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    tmp_path = "{}/{}.tar.gz".format(tmp_dir, output_name)
    print("Start to pack current python env")
    pack_conda_main(conda_name, tmp_path)
    print("Packing has been completed: {}".format(tmp_path))
    return tmp_path


def get_conda_python_path():
    conda_env_path = "/".join(detect_python_location().split("/")[:-2])
    python_interpreters = glob.glob("{}/lib/python*".format(conda_env_path))
    invalidInputError(len(python_interpreters) == 1,
                      "Conda env should contain a single Python "
                      "interpreter, but got: {}".format(python_interpreters))
    return python_interpreters[0]


def get_executor_conda_zoo_classpath(conda_path):
    from bigdl.dllib.utils.engine import get_bigdl_jars
    bigdl_jars = get_bigdl_jars()
    python_interpreter_name = get_conda_python_path().split("/")[-1]  # Python version
    prefix = "{}/lib/{}/site-packages/" \
        .format(conda_path, python_interpreter_name)
    executor_classpath = []
    for jar_path in list(bigdl_jars):
        postfix = "/".join(jar_path.split("/")[-5:])
        executor_classpath.append("{}/{}".format(prefix, postfix))
    return executor_classpath


def get_zoo_bigdl_classpath_on_driver():
    from bigdl.dllib.utils.engine import get_bigdl_classpath
    bigdl_classpath = get_bigdl_classpath()
    invalidInputError(bigdl_classpath,
                      "Cannot find BigDL classpath, please check your installation")
    return bigdl_classpath


def set_python_home():
    if "PYTHONHOME" not in os.environ:
        os.environ['PYTHONHOME'] = "/".join(detect_python_location().split("/")[:-2])


def get_bigdl_class_version():
    from bigdl.dllib.utils.engine import get_bigdl_jars
    bigdl_jars = get_bigdl_jars()
    try:
        bigdl_class_version = re.search('spark_(.+?)-jar', bigdl_jars[1]).group(1)[6:]
    except AttributeError:
        # not found
        bigdl_class_version = 'Cannot find BigDL classpath, please check your installation'
    return bigdl_class_version


def get_bigdl_image_workdir():
    bigdl_image_workdir = "/opt/spark/work-dir"  # WORKDIR defined in dockerfile
    return bigdl_image_workdir


def _is_scalar_type(dtype, accept_str_col=False):
    import pyspark.sql.types as df_types
    if isinstance(dtype, df_types.FloatType):
        return True
    if isinstance(dtype, df_types.IntegerType):
        return True
    if isinstance(dtype, df_types.LongType):
        return True
    if isinstance(dtype, df_types.DoubleType):
        return True
    if isinstance(dtype, df_types.TimestampType):
        return True
    if isinstance(dtype, df_types.DecimalType):
        return True
    if accept_str_col and isinstance(dtype, df_types.StringType):
        return True
    return False


def convert_row_to_numpy(row, schema, feature_cols, label_cols, accept_str_col=False):
    def convert_for_cols(row, cols):
        import pyspark.sql.types as df_types
        result = []
        for name in cols:
            feature_type = schema[name].dataType
            if _is_scalar_type(feature_type, accept_str_col):
                if isinstance(feature_type, df_types.FloatType):
                    result.append(np.array(row[name]).astype(np.float32))
                elif isinstance(feature_type, df_types.DoubleType):
                    result.append(np.array(row[name]).astype(np.float64))
                elif isinstance(feature_type, df_types.TimestampType):
                    result.append(np.array(row[name]).astype('datetime64[ns]'))
                elif isinstance(feature_type, df_types.IntegerType):
                    result.append(np.array(row[name]).astype(np.int32))
                elif isinstance(feature_type, df_types.LongType):
                    result.append(np.array(row[name]).astype(np.int64))
                elif isinstance(feature_type, df_types.DecimalType):
                    result.append(np.array(row[name]).astype(np.float64))
                else:
                    result.append(np.array(row[name]))
            elif isinstance(feature_type, df_types.ArrayType):
                if accept_str_col and isinstance(feature_type.elementType, df_types.StringType):
                    result.append(np.array(row[name]).astype(np.str))
                elif isinstance(feature_type.elementType, df_types.FloatType):
                    result.append(np.array(row[name]).astype(np.float32))
                elif isinstance(feature_type.elementType, df_types.DoubleType):
                    result.append(np.array(row[name]).astype(np.float64))
                elif isinstance(feature_type.elementType, df_types.IntegerType):
                    result.append(np.array(row[name]).astype(np.int32))
                elif isinstance(feature_type.elementType, df_types.LongType):
                    result.append(np.array(row[name]).astype(np.int64))
                elif isinstance(feature_type.elementType, df_types.DecimalType):
                    result.append(np.array(row[name]).astype(np.float64))
                else:
                    result.append(np.array(row[name]))
            elif isinstance(row[name], DenseVector):
                result.append(row[name].values.astype(np.float32))
            else:
                invalidInputError(isinstance(row[name], SparseVector),
                                  "unsupported field {}, data {}".format(schema[name], row[name]))
                result.append(row[name].toArray())
        if len(result) == 1:
            return result[0]
        return result

    features = convert_for_cols(row, feature_cols)
    if label_cols:
        labels = convert_for_cols(row, label_cols)
        return (features, labels)
    else:
        return (features,)


def toMultiShape(shape):
    if any(isinstance(i, list) for i in shape):  # multi shape
        return shape
    elif any(isinstance(i, tuple) for i in shape):
        return [list(s) for s in shape]
    elif isinstance(shape, tuple):
        return [list(shape)]
    else:
        return [shape]


# TODO: create a shape mapping here.
def remove_batch(shape):
    if any(isinstance(i, list) or isinstance(i, tuple) for i in shape):  # multi shape
        return [remove_batch(s) for s in shape]
    else:
        return list(shape[1:])
