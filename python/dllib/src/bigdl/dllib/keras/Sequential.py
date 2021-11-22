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

from bigdl.dllib.utils.utils import remove_batch
from .engine.topology import KerasNet
from bigdl.dllib.utils.common import to_list
from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.utils.dllib_export import keras_export

if sys.version >= '3':
    long = int
    unicode = str

@keras_export('bigdl.dllib.keras.Sequential')
class Sequential(KerasNet):
    """
    Container for a sequential model.

    # Arguments
    name: String to specify the name of the sequential model. Default is None.

    >>> sequential = Sequential(name="seq1")
    creating: createZooKerasSequential
    """

    def __init__(self, jvalue=None, **kwargs):
        super(Sequential, self).__init__(jvalue, **kwargs)

    # TODO: expose is_built from scala side
    def is_built(self):
        try:
            self.get_output_shape()
            return True
        except:
            return False

    def add(self, model):
        from bigdl.dllib.keras.autograd import Lambda
        if (isinstance(model, Lambda)):
            if not self.is_built():
                if not model.input_shape:
                    raise Exception("You should specify inputShape for the first layer")
                input_shapes = model.input_shape
            else:
                input_shapes = self.get_output_shape()
            model = model.create(remove_batch(input_shapes))
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