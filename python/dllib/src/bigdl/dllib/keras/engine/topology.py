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

from zoo.pipeline.api.keras.layers.layer import ZooKerasCreator
from bigdl.nn.keras.topology import KerasModel as BKerasModel
from bigdl.util.common import to_list


class ZooKerasModel(ZooKerasCreator, BKerasModel):
    pass


class Sequential(ZooKerasModel):
    """
    Container for a sequential model.

    # Arguments
    name: String to specify the name of the sequential model. Default is None.

    >>> sequential = Sequential(name="seq1")
    creating: createZooKerasSequential
    """
    def __init__(self, jvalue=None, **kwargs):
        super(Sequential, self).__init__(jvalue, **kwargs)

    def add(self, model):
        self.value.add(model.value)
        return self

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        """
        Create a Python Model base on the given java value
        :param jvalue: Java object create by Py4j
        :return: A Python Model
        """
        model = Sequential(jvalue=jvalue)
        model.value = jvalue
        return model


class Model(ZooKerasModel):
    """
    Container for a graph model.

    # Arguments
    input: An input node or a list of input nodes.
    output: An output node or a list of output nodes.
    name: String to specify the name of the graph model. Default is None.
    """
    def __init__(self, input, output, jvalue=None, **kwargs):
        super(Model, self).__init__(jvalue,
                                    to_list(input),
                                    to_list(output),
                                    **kwargs)

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        """
        Create a Python Model base on the given java value
        :param jvalue: Java object create by Py4j
        :return: A Python Model
        """
        model = Model([], [], jvalue=jvalue)
        model.value = jvalue
        return model
