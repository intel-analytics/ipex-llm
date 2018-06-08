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

from bigdl.nn.keras.layer import KerasLayer
from zoo.pipeline.api.keras.utils import *

if sys.version >= '3':
    long = int
    unicode = str


class ZooKerasCreator(JavaValue):
    def jvm_class_constructor(self):
        name = "createZooKeras" + self.__class__.__name__
        print("creating: " + name)
        return name


class ZooKerasLayer(ZooKerasCreator, KerasLayer):
    @classmethod
    def of(cls, jvalue, bigdl_type="float"):
        return KerasLayer(jvalue, bigdl_type)

    def __call__(self, x):
        """
        Some other modules point to current module
        :param x: input variables. x is either a Variable or list of Variable.
        :return: Variable containing current module
        """
        from zoo.pipeline.api.autograd import Variable
        return Variable.from_jvalue(callBigDlFunc(self.bigdl_type,
                                                  "connectInputs",
                                                  self,
                                                  to_list(x)))
