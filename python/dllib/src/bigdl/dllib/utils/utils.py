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

import glob
from zoo.common import Sample


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
    Ray is not involved, calling ray.services.get_node_ip_address would introduce Ray overhead.
    """
    import socket
    import errno
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This command will raise an exception if there is no internet connection.
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
        raise EnvironmentError(err +
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
    raise EnvironmentError(err + "Failed to detect the current conda environment. Please verify"
                                 "your conda installation and activate the env you want to use")


# This is adopted from conda-pack.
def pack_conda_main(conda_name, tmp_path):
    import subprocess
    import os
    pack_env = os.environ.copy()
    if "PYTHONHOME" in pack_env:
        pack_env.pop("PYTHONHOME")
    pack_cmd = "conda pack --format tar.gz --n-threads 8 -f -n {} -o {}"\
        .format(conda_name, tmp_path)
    pro = subprocess.Popen(pack_cmd, shell=True, env=pack_env)
    os.waitpid(pro.pid, 0)


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
    assert len(python_interpreters) == 1, "Conda env should contain a single Python " \
                                          "interpreter, but got: {}".format(python_interpreters)
    return python_interpreters[0]


def get_executor_conda_zoo_classpath(conda_path):
    zoo_classpath, bigdl_classpath = get_zoo_bigdl_classpath_on_driver()
    zoo_jar_name = zoo_classpath.split("/")[-1]
    bigdl_jar_name = bigdl_classpath.split("/")[-1]
    python_interpreter_name = get_conda_python_path().split("/")[-1]  # Python version
    prefix = "{}/lib/{}/site-packages/"\
        .format(conda_path, python_interpreter_name)
    return ["{}/zoo/share/lib/{}".format(prefix, zoo_jar_name),
            "{}/bigdl/share/lib/{}".format(prefix, bigdl_jar_name)]


def get_zoo_bigdl_classpath_on_driver():
    from bigdl.util.engine import get_bigdl_classpath
    from zoo.util.engine import get_analytics_zoo_classpath
    bigdl_classpath = get_bigdl_classpath()
    assert bigdl_classpath, "Cannot find BigDL classpath, please check your installation"
    zoo_classpath = get_analytics_zoo_classpath()
    assert zoo_classpath, "Cannot find Analytics-Zoo classpath, please check your installation"
    return zoo_classpath, bigdl_classpath


def set_python_home():
    import os
    if "PYTHONHOME" not in os.environ:
        os.environ['PYTHONHOME'] = "/".join(detect_python_location().split("/")[:-2])
