#
# Licensed to Intel Corporation under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# Intel Corporation licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from util.common import callBigDlFunc
from util.common import JavaValue
from util.common import callJavaFunc
from pyspark import SparkContext
import numpy as np
import os
from distutils.dir_util import mkpath


import sys
if sys.version >= '3':
    long = int
    unicode = str


class Validator(JavaValue):
    def __init__(self,  model, val_rdd, batch_size, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, model, val_rdd, batch_size)

    def test(self, val_methods):
        return callBigDlFunc(self.bigdl_type, "test", self.value, val_methods)


class MaxEpoch(JavaValue):
    def __init__(self, max_epoch, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, max_epoch)


class EveryEpoch(JavaValue):
    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type)


class SeveralIteration(JavaValue):
    def __init__(self, interval, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, interval)


class MaxIteration(JavaValue):
    def __init__(self, max, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, max)


class ClassNLLCriterion(JavaValue):
    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type)


class MSECriterion(JavaValue):
    def __init__(self, bigdl_type="float"):
            JavaValue.__init__(self, None, bigdl_type)


class Optimizer(JavaValue):

    def __init__(self,
                 model,
                 training_rdd,
                 criterion,
                 end_trigger,
                 batch_size,
                 optim_method="SGD",
                 state={},
                 bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, model.value,
                           training_rdd, criterion, optim_method,
                           state, end_trigger, batch_size)

    def setvalidation(self, batch_size, val_rdd, trigger, val_method=["top1"]):
        callBigDlFunc(self.bigdl_type, "setValidation", self.value, batch_size,
                      trigger, val_rdd, val_method)

    def setcheckpoint(self, checkpoint_trigger,
                      checkpoint_path, isOverWrite=True):
        if not os.path.exists(checkpoint_path):
            mkpath(checkpoint_path)
        callBigDlFunc(self.bigdl_type, "setCheckPoint", self.value,
                      checkpoint_trigger, checkpoint_path, isOverWrite)

    # return a module
    def optimize(self):
        jmodel = callJavaFunc(SparkContext.getOrCreate(), self.value.optimize)
        from nn.layer import Model
        return Model.of(jmodel)


def _test():
    import doctest
    from pyspark import SparkContext
    from optim import optimizer
    globs = optimizer.__dict__.copy()
    sc = SparkContext(master="local[4]", appName="test optimizer")
    globs['sc'] = sc
    (failure_count, test_count) = doctest.testmod(globs=globs,
                                                  optionflags=doctest.ELLIPSIS)
    if failure_count:
        exit(-1)

if __name__ == "__main__":
    _test()
