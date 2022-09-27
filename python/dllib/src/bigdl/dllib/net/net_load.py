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

import importlib
import os
import sys

from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.nn.layer import Model as BModel
from bigdl.dllib.net.graph_net import GraphNet
from bigdl.dllib.utils.log4Error import *


if sys.version >= '3':
    long = int
    unicode = str


class JavaToPython:
    # TODO: Add more mapping here as it only support Model and Sequential for now.
    def __init__(self, jvalue, bigdl_type="float"):
        self.jvaule = jvalue
        self.jfullname = callZooFunc(bigdl_type,
                                     "getRealClassNameOfJValue",
                                     jvalue)

    def get_python_class(self):
        """
        Redirect the jvalue to the proper python class.
        :param jvalue: Java object create by Py4j
        :return: A proper Python wrapper which would be a Model, Sequential...
        """

        jpackage_name = ".".join(self.jfullname.split(".")[:-1])
        pclass_name = self._get_py_name(self.jfullname.split(".")[-1])
        if self.jfullname in ("com.intel.analytics.bigdl.dllib.keras.Sequential",
                              "com.intel.analytics.bigdl.dllib.keras.Model"):
            base_module = self._load_ppackage_by_jpackage(self.jfullname)
        else:
            base_module = self._load_ppackage_by_jpackage(jpackage_name)
        if pclass_name in dir(base_module):
            pclass = getattr(base_module, pclass_name)
            invalidInputError("from_jvalue" in dir(pclass),
                              "pclass: {} should implement from_jvalue method".format(pclass))
            return pclass
        invalidInputError(False, "No proper python class for: {}".format(self.jfullname))

    def _get_py_name(self, jclass_name):
        if jclass_name == "Model":
            return "Model"
        elif jclass_name == "Sequential":
            return "Sequential"
        else:
            invalidInputError(False, "Not supported type: {}".format(jclass_name))

    def _load_ppackage_by_jpackage(self, jpackage_name):
        if jpackage_name in ("com.intel.analytics.bigdl.dllib.keras.Model",
                             "com.intel.analytics.bigdl.dllib.keras.Sequential"):
            return importlib.import_module('bigdl.dllib.keras.models')
        invalidInputError(False, "Not supported package: {}".format(jpackage_name))


class Net:

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        pclass = JavaToPython(jvalue).get_python_class()
        return getattr(pclass, "from_jvalue")(jvalue, bigdl_type)

    @staticmethod
    def load_bigdl(model_path, weight_path=None, bigdl_type="float"):
        """
        Load a pre-trained BigDL model.

        :param model_path: The path to the pre-trained model.
        :param weight_path: The path to the weights of the pre-trained model. Default is None.
        :return: A pre-trained model.
        """
        jmodel = callZooFunc(bigdl_type, "netLoadBigDL", model_path, weight_path)
        return GraphNet.from_jvalue(jmodel)

    @staticmethod
    def load(model_path, weight_path=None, bigdl_type="float"):
        """
        Load an existing BigDL model defined in Keras-style(with weights).

        :param model_path: The path to load the saved model.
                          Local file system, HDFS and Amazon S3 are supported.
                          HDFS path should be like 'hdfs://[host]:[port]/xxx'.
                          Amazon S3 path should be like 's3a://bucket/xxx'.
        :param weight_path: The path for pre-trained weights if any. Default is None.
        :return: A BigDL model.
        """
        jmodel = callZooFunc(bigdl_type, "netLoad", model_path, weight_path)
        return Net.from_jvalue(jmodel, bigdl_type)

    @staticmethod
    def load_torch(path, bigdl_type="float"):
        """
        Load a pre-trained Torch model.

        :param path: The path containing the pre-trained model.
        :return: A pre-trained model.
        """
        jmodel = callZooFunc(bigdl_type, "netLoadTorch", path)
        return GraphNet.from_jvalue(jmodel, bigdl_type)

    @staticmethod
    def load_caffe(def_path, model_path, bigdl_type="float"):
        """
        Load a pre-trained Caffe model.

        :param def_path: The path containing the caffe model definition.
        :param model_path: The path containing the pre-trained caffe model.
        :return: A pre-trained model.
        """
        jmodel = callZooFunc(bigdl_type, "netLoadCaffe", def_path, model_path)
        return GraphNet.from_jvalue(jmodel, bigdl_type)

    @staticmethod
    def load_keras(json_path=None, hdf5_path=None, by_name=False):
        """
        Load a pre-trained Keras model.

        :param json_path: The json path containing the keras model definition. Default is None.
        :param hdf5_path: The HDF5 path containing the pre-trained keras model weights
                        with or without the model architecture. Default is None.
        :param by_name: by default the architecture should be unchanged.
                        If set as True, only layers with the same name will be loaded.
        :return: A BigDL model.
        """
        return BModel.load_keras(json_path, hdf5_path, by_name)
