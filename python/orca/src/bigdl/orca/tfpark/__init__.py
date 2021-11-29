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


def check_tf_version():
    import logging
    try:
        import tensorflow as tf
    except Exception as e:
        return False, RuntimeError("Importing TensorFlow failed, "
                                   "please install tensorflow 1.15.0.", e)

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
            return False, RuntimeError(message)
    return True, None

passed, error = check_tf_version()

if passed:
    from .model import KerasModel
    from .estimator import TFEstimator
    from .tf_optimizer import TFOptimizer
    from .tf_dataset import TFDataset
    from .zoo_optimizer import ZooOptimizer
    from .tf_predictor import TFPredictor
    from .tfnet import TFNet
else:
    CLASSES_WITH_MAGIC_METHODS = (str(), object, float(), dict())

    # Combines all magic methods I can think of.
    MAGIC_METHODS_TO_CHANGE = set()
    for i in CLASSES_WITH_MAGIC_METHODS:
        MAGIC_METHODS_TO_CHANGE |= set(dir(i))
    MAGIC_METHODS_TO_CHANGE.add('__call__')
    # __init__ and __new__ must not raise an UnusableObjectError
    # otherwise it would raise error even on creation of objects.
    MAGIC_METHODS_TO_CHANGE -= {'__class__', '__init__', '__new__'}

    def error_func(*args, **kwargs):
        """(nearly) all magic methods will be set to this function."""
        raise error

    class UnusableClass(object):
        def __init__(*args, **kwargs):
            pass

    for i in MAGIC_METHODS_TO_CHANGE:
        setattr(UnusableClass, i, error_func)

    KerasModel = UnusableClass
    TFEstimator = UnusableClass
    TFOptimizer = UnusableClass
    TFDataset = UnusableClass
    ZooOptimizer = UnusableClass
    TFPredictor = UnusableClass
    TFNet = UnusableClass
