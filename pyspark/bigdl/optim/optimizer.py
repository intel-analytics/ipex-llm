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


import os
import sys
from distutils.dir_util import mkpath

from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from bigdl.util.common import callJavaFunc
from bigdl.util.common import get_spark_context

if sys.version >= '3':
    long = int
    unicode = str


class MaxIteration(JavaValue):
    """
    A trigger specifies a timespot or several timespots during training, 
    and a corresponding action will be taken when the timespot(s) is reached.  
    MaxIteration is a trigger that triggers an action when training reaches 
    the number of iterations specified by "max". 
    Usually used as end_trigger when creating an Optimizer. 

    >>> maxIteration = MaxIteration(20)
    creating: createMaxIteration
    """
    def __init__(self, max, bigdl_type="float"):
        """
        Create a MaxIteration trigger.

        :param max: max
        """
        JavaValue.__init__(self, None, bigdl_type, max)


class MaxEpoch(JavaValue):
    """
    A trigger specifies a timespot or several timespots during training,
    and a corresponding action will be taken when the timespot(s) is reached.
    MaxEpoch is a trigger that triggers an action when training reaches
    the number of epochs specified by "max_epoch".
    Usually used as end_trigger when creating an Optimizer.

    >>> maxEpoch = MaxEpoch(2)
    creating: createMaxEpoch
    """
    def __init__(self, max_epoch, bigdl_type="float"):
        """
        Create a MaxEpoch trigger.
 
        :param max_epoch: max_epoch
        """
        JavaValue.__init__(self, None, bigdl_type, max_epoch)


class EveryEpoch(JavaValue):
    """
    A trigger specifies a timespot or several timespots during training,
    and a corresponding action will be taken when the timespot(s) is reached.
    EveryEpoch is a trigger that triggers an action when each epoch finishs.
    Could be used as trigger in setvalidation and setcheckpoint in Optimizer,
    and also in TrainSummary.set_summary_trigger.

    >>> everyEpoch = EveryEpoch()
    creating: createEveryEpoch
    """
    def __init__(self, bigdl_type="float"):
        """
        Create a EveryEpoch trigger.     
        """
        JavaValue.__init__(self, None, bigdl_type)


class SeveralIteration(JavaValue):
    """
    A trigger specifies a timespot or several timespots during training,
    and a corresponding action will be taken when the timespot(s) is reached.
    SeveralIteration is a trigger that triggers an action every "n" 
    iterations.
    Could be used as trigger in setvalidation and setcheckpoint in Optimizer,
    and also in TrainSummary.set_summary_trigger.

    >>> serveralIteration = SeveralIteration(2)
    creating: createSeveralIteration
    """
    def __init__(self, interval, bigdl_type="float"):
        """
        Create a SeveralIteration trigger.

        :param interval: interval is the "n" where an action is triggered 
           every "n" iterations
        """
        JavaValue.__init__(self, None, bigdl_type, interval)


class Poly(JavaValue):
    """
    A learning rate decay policy, where the effective learning rate 
    follows a polynomial decay, to be zero by the max_iteration. 
    Calculation: base_lr (1 - iter/max_iteration) ^ (power)
   
    :param power 
    :param max_iteration

    >>> poly = Poly(0.5, 2)
    creating: createPoly
    """
    def __init__(self, power, max_iteration, bigdl_type="float"):
            JavaValue.__init__(self, None, bigdl_type, power, max_iteration)


class Step(JavaValue):
    """
    A learning rate decay policy, where the effective learning rate is 
    calculated as base_lr * gamma ^ (floor(iter / step_size))
   
    :param step_size
    :param gamma

    >>> step = Step(2, 0.3)
    creating: createStep
    """
    def __init__(self, step_size, gamma, bigdl_type="float"):
            JavaValue.__init__(self, None, bigdl_type, step_size, gamma)


class Optimizer(JavaValue):
    """
    An optimizer is in general to minimize any function with respect 
    to a set of parameters. In case of training a neural network, 
    an optimizer tries to minimize the loss of the neural net with 
    respect to its weights/biases, over the training set. 
    """
    def __init__(self,
                 model,
                 training_rdd,
                 criterion,
                 end_trigger,
                 batch_size,
                 optim_method="SGD",
                 state={},
                 bigdl_type="float"):
       """
       Create an optimizer.

       :param model: the neural net model
       :param traiing_rdd: the training dataset 
       :param criterion: the loss function
       :param optim_method: the algorithm to use for optimization, 
          e.g. SGD, Adagrad, etc.
       :param state: a set of initial configurations for optimizer, 
          provided as a dict e.g. configurable params include: 
          learningRate, learningRateDecay,etc.  
       :param end_trigger: when to end the optimization
       :param batch_size: training batch size
       """ 
       JavaValue.__init__(self, None, bigdl_type, model.value,
                           training_rdd, criterion, optim_method,
                           state, end_trigger, batch_size)

    def set_validation(self, batch_size, val_rdd, trigger, val_method=["Top1Accuracy"]):
        """
        Configure validation settings. 

        :param batch_size: validation batch size
        :param val_rdd: validation dataset
        :param trigger: validation interval
        :param val_method: the ValidationMethod to use,
           e.g. "Top1Accuracy", "Top5Accuracy", "Loss"
        """
        callBigDlFunc(self.bigdl_type, "setValidation", self.value, batch_size,
                      trigger, val_rdd, val_method)

    def set_model(self, model):
        """
        Set model. 

        :param model: new model
        """
        self.value.setModel(model.value)

    def set_checkpoint(self, checkpoint_trigger,
                      checkpoint_path, isOverWrite=True):
        """
        Configure checkpoint settings. 

        :param checkpoint_trigger: the interval to write snapshots
        :param checkpoint_path: the path to write snapshots into
        :param isOverWrite: whether to overwrite existing snapshots in path. 
           default is True
        """
        if not os.path.exists(checkpoint_path):
            mkpath(checkpoint_path)
        callBigDlFunc(self.bigdl_type, "setCheckPoint", self.value,
                      checkpoint_trigger, checkpoint_path, isOverWrite)

    # return a module
    def optimize(self):
        """
        Do an optimization. 
        """
        jmodel = callJavaFunc(get_spark_context(), self.value.optimize)
        from bigdl.nn.layer import Model
        return Model.of(jmodel)

    def set_train_summary(self, summary):
        """
        Set train summary. A TrainSummary object contains information 
        necesary for the optimizer to know how often the logs are recorded, 
        where to store the logs and how to retrieve them, etc. For details, 
        refer to the docs of TrainSummary.

        :param summary: a TrainSummary object
        """
        callBigDlFunc(self.bigdl_type, "setTrainSummary", self.value,
                      summary)
        return self

    def set_val_summary(self, summary):
        """
        Set validation summary. A ValidationSummary object contains information 
        necesary for the optimizer to know how often the logs are recorded, 
        where to store the logs and how to retrieve them, etc. For details, 
        refer to the docs of ValidationSummary.

        :param summary: a ValidationSummary object

        """
        callBigDlFunc(self.bigdl_type, "setValSummary", self.value,
                      summary)
        return self

    def prepare_input(self):
        """
        Load input. Notebook user can call this method to seprate load data and
        create optimizer time 
        """
        print("Loading input ...")
        self.value.prepareInput()


class TrainSummary(JavaValue, ):
    """
    A logging facility which allows user to trace how indicators (e.g. 
    learning rate, training loss, throughput, etc.) change with iterations/time 
    in an optimization process. TrainSummary is for training indicators only 
    (check ValidationSummary for validation indicators).  It contains necessary 
    information for the optimizer to know where to store the logs, how to 
    retrieve the logs, and so on. - The logs are written in tensorflow-compatible 
    format so that they can be visualized directly using tensorboard. Also the 
    logs can be retrieved as ndarrays and visualized using python libraries 
    such as matplotlib (in notebook, etc.). 
    
    Use optimizer.setTrainSummary to enable train logger. 
    """
    def __init__(self, log_dir, app_name, bigdl_type="float"):
        """
        Create a TrainSummary. Logs will be saved to log_dir/app_name/train.

        :param log_dir: the root dir to store the logs
        :param app_name: the application name
        """
        JavaValue.__init__(self, None, bigdl_type, log_dir, app_name)

    def read_scalar(self, tag):
        """
        Retrieve train logs by type. Return an array of records in the format 
        (step,value,wallClockTime). - "Step" is the iteration count by default.
        
        :param tag: the type of the logs, Supported tags are: "LearningRate", 
           "Loss", "Throughput"
        """
        return callBigDlFunc(self.bigdl_type, "summaryReadScalar", self.value,
                             tag)

    def set_summary_trigger(self, name, trigger):
        """
        Set the interval of recording for each indicator. 
        
        :param tag: tag name. Supported tag names are "LearningRate", "Loss", 
           "Throughput", "Parameters". "Parameters" is an umbrella tag that 
           includes weight, bias, gradWeight, gradBias, and some running status
           (eg. runningMean and runningVar in BatchNormalization). If you 
           didn't set any triggers, we will by default record Loss and Throughput
           in each iteration, while *NOT* recording LearningRate and Parameters,
           as recording parameters may introduce substantial overhead when the
           model is very big, LearningRate is not a public attribute for all
           OptimMethod.
        :param trigger: trigger
        """
        return callBigDlFunc(self.bigdl_type, "summarySetTrigger", self.value,
                             name, trigger)


class ValidationSummary(JavaValue):
    """
     A logging facility which allows user to trace how indicators (e.g. 
     validation loss, top1 accuray, top5 accuracy etc.) change with 
     iterations/time in an optimization process. ValidationSummary is for 
     validation indicators only (check TrainSummary for train indicators).  
     It contains necessary information for the optimizer to know where to 
     store the logs, how to retrieve the logs, and so on. - The logs are 
     written in tensorflow-compatible format so that they can be visualized 
     directly using tensorboard. Also the logs can be retrieved as ndarrays 
     and visualized using python libraries such as matplotlib 
     (in notebook, etc.).
    
     Use optimizer.setValidationSummary to enable validation logger.
    """
    def __init__(self, log_dir, app_name, bigdl_type="float"):
        """
        Create a ValidationSummary. Logs will be saved to 
        log_dir/app_name/train. By default, all ValidationMethod set into 
        optimizer will be recorded and the recording interval is the same 
        as trigger of ValidationMethod in the optimizer. 

        :param log_dir: the root dir to store the logs
        :param app_name: the application name
        """
        JavaValue.__init__(self, None, bigdl_type, log_dir, app_name)

    def read_scalar(self, tag):
        """
        Retrieve validation logs by type. Return an array of records in the 
        format (step,value,wallClockTime). - "Step" is the iteration count 
        by default.

        :param tag: the type of the logs. The tag should match the name of 
           the ValidationMethod set into the optimizer. e.g. 
           "Top1AccuracyLoss","Top1Accuracy" or "Top5Accuracy".
        """
        return callBigDlFunc(self.bigdl_type, "summaryReadScalar", self.value,
                             tag)


def _test():
    import doctest
    from pyspark import SparkContext
    from bigdl.optim import optimizer
    from bigdl.util.common import init_engine
    from bigdl.util.common import create_spark_conf
    globs = optimizer.__dict__.copy()
    sc = SparkContext(master="local[4]", appName="test optimizer",
                      conf=create_spark_conf())
    init_engine()
    globs['sc'] = sc
    (failure_count, test_count) = doctest.testmod(globs=globs,
                                                  optionflags=doctest.ELLIPSIS)
    if failure_count:
        exit(-1)

if __name__ == "__main__":
    _test()
