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
from util.common import callBigDlFunc
from util.common import JavaValue
from util.common import callJavaFunc
from pyspark import SparkContext

import numpy as np

if sys.version >= '3':
    long = int
    unicode = str


class Criterion(JavaValue):
    def __init__(self, jvalue, bigdl_type, *args):
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), *args)
        self.bigdl_type = bigdl_type

    @classmethod
    def of(cls, jcriterion, bigdl_type="float"):
        criterion = Criterion(bigdl_type, jcriterion)
        criterion.value = jcriterion
        criterion.bigdl_type = bigdl_type
        return criterion


class ClassNLLCriterion(Criterion):
    '''
    >>> classNLLCriterion = ClassNLLCriterion()
    creating: createClassNLLCriterion
    '''
    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type)


class MSECriterion(Criterion):
    '''
    >>> mSECriterion = MSECriterion()
    creating: createMSECriterion
    '''
    def __init__(self, bigdl_type="float"):
            JavaValue.__init__(self, None, bigdl_type)
