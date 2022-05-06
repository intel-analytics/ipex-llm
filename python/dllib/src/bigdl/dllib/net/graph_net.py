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

from bigdl.dllib.nn.layer import Model as BModel
from bigdl.dllib.feature.image import ImageSet
from bigdl.dllib.feature.text import TextSet
from bigdl.dllib.keras.base import ZooKerasLayer
from bigdl.dllib.keras.utils import *
from bigdl.dllib.nn.layer import Layer
from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.utils.log4Error import *


if sys.version >= '3':
    long = int
    unicode = str


class GraphNet(BModel):
    def __init__(self, input, output, jvalue=None, bigdl_type="float", **kwargs):
        super(BModel, self).__init__(jvalue,
                                     to_list(input),
                                     to_list(output),
                                     bigdl_type,
                                     **kwargs)

    def predict(self, x, batch_per_thread=4, distributed=True):
        """
        Use a model to do prediction.

        # Arguments
        x: Prediction data. A Numpy array or RDD of Sample or ImageSet.
        batch_per_thread:
          The default value is 4.
          When distributed is True,the total batch size is batch_per_thread * rdd.getNumPartitions.
          When distributed is False the total batch size is batch_per_thread * numOfCores.
        distributed: Boolean. Whether to do prediction in distributed mode or local mode.
                     Default is True. In local mode, x must be a Numpy array.
        """
        if isinstance(x, ImageSet) or isinstance(x, TextSet):
            results = callZooFunc(self.bigdl_type, "zooPredict",
                                  self.value,
                                  x,
                                  batch_per_thread)
            return ImageSet(results) if isinstance(x, ImageSet) else TextSet(results)
        if distributed:
            if isinstance(x, np.ndarray):
                data_rdd = to_sample_rdd(x, np.zeros([x.shape[0]]))
            elif isinstance(x, RDD):
                data_rdd = x
            else:
                invalidInputError(False, "Unsupported prediction data type: %s" % type(x))
            results = callZooFunc(self.bigdl_type, "zooPredict",
                                  self.value,
                                  data_rdd,
                                  batch_per_thread)
            return results.map(lambda result: Layer.convert_output(result))
        else:
            if isinstance(x, np.ndarray) or isinstance(x, list):
                results = callZooFunc(self.bigdl_type, "zooPredict",
                                      self.value,
                                      self._to_jtensors(x),
                                      batch_per_thread)
                return [Layer.convert_output(result) for result in results]
            else:
                invalidInputError(False, "Unsupported prediction data type: %s" % type(x))

    def flattened_layers(self, include_container=False):
        jlayers = callZooFunc(self.bigdl_type, "getFlattenSubModules", self, include_container)
        layers = [Layer.of(jlayer) for jlayer in jlayers]
        return layers

    @property
    def layers(self):
        jlayers = callZooFunc(self.bigdl_type, "getSubModules", self)
        layers = [Layer.of(jlayer) for jlayer in jlayers]
        return layers

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        """
        Create a Python Model base on the given java value

        :param jvalue: Java object create by Py4j
        :return: A Python Model
        """
        model = GraphNet([], [], jvalue=jvalue, bigdl_type=bigdl_type)
        model.value = jvalue
        return model

    def new_graph(self, outputs):
        """
        Specify a list of nodes as output and return a new graph using the existing nodes

        :param outputs: A list of nodes specified
        :return: A graph model
        """
        value = callZooFunc(self.bigdl_type, "newGraph", self.value, outputs)
        return self.from_jvalue(value, self.bigdl_type)

    def freeze_up_to(self, names):
        """
        Freeze the model from the bottom up to the layers specified by names (inclusive).
        This is useful for finetuning a model

        :param names: A list of module names to be Freezed
        :return: current graph model
        """
        callZooFunc(self.bigdl_type, "freezeUpTo", self.value, names)

    def unfreeze(self, names=None):
        """
        "unfreeze" module, i.e. make the module parameters(weight/bias, if exists)
        to be trained(updated) in training process.
        If 'names' is a non-empty list, unfreeze layers that match given names

        :param names: list of module names to be unFreezed. Default is None.
        :return: current graph model
        """
        callZooFunc(self.bigdl_type, "unFreeze", self.value, names)

    def to_keras(self):
        value = callZooFunc(self.bigdl_type, "netToKeras", self.value)
        return ZooKerasLayer.of(value, self.bigdl_type)
