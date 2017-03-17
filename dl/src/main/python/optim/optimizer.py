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


class MaxEpoch(JavaValue):
    '''
    >>> maxEpoch = MaxEpoch(2)
    creating: createMaxEpoch
    '''
    def __init__(self, max_epoch, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, max_epoch)


class EveryEpoch(JavaValue):
    '''
    >>> everyEpoch = EveryEpoch()
    creating: createEveryEpoch
    '''
    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type)


class SeveralIteration(JavaValue):
    '''
    >>> serveralIteration = SeveralIteration(2)
    creating: createSeveralIteration
    '''
    def __init__(self, interval, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, interval)


class MaxIteration(JavaValue):
    '''
    >>> maxIteration = MaxIteration(20)
    creating: createMaxIteration
    '''
    def __init__(self, max, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, max)


class ClassNLLCriterion(JavaValue):
    '''
    >>> classNLLCriterion = ClassNLLCriterion()
    creating: createClassNLLCriterion
    '''
    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type)


class MSECriterion(JavaValue):
    '''
    >>> mSECriterion = MSECriterion()
    creating: createMSECriterion
    '''
    def __init__(self, bigdl_type="float"):
            JavaValue.__init__(self, None, bigdl_type)


class Poly(JavaValue):
    '''
    >>> poly = Poly(0.5, 2)
    creating: createPoly
    '''
    def __init__(self, power, max_iteration, bigdl_type="float"):
            JavaValue.__init__(self, None, bigdl_type, power, max_iteration)


class Step(JavaValue):
    '''
    >>> step = Step(2, 0.3)
    creating: createStep
    '''
    def __init__(self, step_size, gamma, bigdl_type="float"):
            JavaValue.__init__(self, None, bigdl_type, step_size, gamma)


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

    def setvalidation(self, batch_size, val_rdd, trigger, val_method=["Top1Accuracy"]):
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

    def set_train_summary(self, summary):
        callBigDlFunc(self.bigdl_type, "setTrainSummary", self.value,
                      summary)
        return self

    def set_val_summary(self, summary):
        callBigDlFunc(self.bigdl_type, "setValSummary", self.value,
                      summary)
        return self


class TrainSummary(JavaValue, ):
    def __init__(self, log_dir, app_name, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, log_dir, app_name)

    def read_scalar(self, tag):
        return callBigDlFunc(self.bigdl_type, "summaryReadScalar", self.value,
                             tag)

    def set_summary_trigger(self, name, trigger):
        return callBigDlFunc(self.bigdl_type, "summarySetTrigger", self.value,
                             name, trigger)


class ValidationSummary(JavaValue):
    def __init__(self, log_dir, app_name, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, log_dir, app_name)

    def read_scalar(self, tag):
        return callBigDlFunc(self.bigdl_type, "summaryReadScalar", self.value,
                             tag)


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
