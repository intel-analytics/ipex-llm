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


def check_tf_version():
    import logging
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("Importing TensorFlow failed, please install tensorflow 1.15.0.", e)

    v_str = tf.__version__
    major, minor, patch = v_str.split(".")
    if v_str != "1.15.0":
        if int(major) == 1:
            logging.warning("\n######################### WARNING ##########################\n"
                            "\nAnalytics Zoo TFPark has only been tested on TensorFlow 1.15.0,"
                            " but your current TensorFlow installation is {}.".format(v_str) +
                            "\nYou may encounter some version incompatibility issues. "
                            "\n##############################################################")
        else:
            message = "Currently Analytics Zoo TFPark only supports TensorFlow 1.15.0, " + \
                      "but your current TensorFlow installation is {}".format(v_str)
            raise RuntimeError(message)

check_tf_version()

from .model import KerasModel
from .estimator import TFEstimator
from .tf_optimizer import TFOptimizer
from .tf_dataset import TFDataset
from .zoo_optimizer import ZooOptimizer
from .tf_predictor import TFPredictor
from .tfnet import TFNet
