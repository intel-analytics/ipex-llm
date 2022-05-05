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


import multiprocessing
import os
import sys
from distutils.dir_util import mkpath

from py4j.java_gateway import JavaObject
from pyspark.rdd import RDD

from bigdl.dllib.utils.common import DOUBLEMAX
from bigdl.dllib.utils.common import JTensor
from bigdl.dllib.utils.common import JavaValue
from bigdl.dllib.utils.common import callBigDlFunc
from bigdl.dllib.utils.common import callJavaFunc
from bigdl.dllib.utils.common import get_node_and_core_number
from bigdl.dllib.utils.common import init_engine
from bigdl.dllib.utils.common import to_list
from bigdl.dllib.feature.dataset.dataset import *
import warnings
from bigdl.dllib.utils.log4Error import *


if sys.version >= '3':
    long = int
    unicode = str


class Top1Accuracy(JavaValue):
    """
    Caculate the percentage that output's max probability index equals target.

    >>> top1 = Top1Accuracy()
    creating: createTop1Accuracy
    """

    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type)


class TreeNNAccuracy(JavaValue):
    """
    Caculate the percentage that output's max probability index equals target.

    >>> top1 = TreeNNAccuracy()
    creating: createTreeNNAccuracy
    """

    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type)


class Top5Accuracy(JavaValue):
    """
    Caculate the percentage that output's max probability index equals target.

    >>> top5 = Top5Accuracy()
    creating: createTop5Accuracy
    """

    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type)


class MeanAveragePrecision(JavaValue):
    """
    Calculate the Mean Average Precision for top-k confident predictions.
    The algorithm follows VOC Challenge after 2007

    >>> MAP = MeanAveragePrecision(10, 20)
    creating: createMeanAveragePrecision
    """

    def __init__(self, k, classes, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, k, classes)


class MeanAveragePrecisionObjectDetection(JavaValue):
    """
    Calculate the Mean Average Precision for Object Detection.

    >>> MAPObj = MeanAveragePrecisionObjectDetection(20)
    creating: createMeanAveragePrecisionObjectDetection
    """

    def __init__(self, classes, iou=0.5, use_voc2007=False, skip_class=-1, bigdl_type="float"):
        """
        :param classes: the number of classes
        :param iou: the IOU threshold
        :param use_voc2007: use validation method before voc2010 (i.e. voc2007)
        :param skip_class: skip calculation on a specific class (e.g. background)
        """
        JavaValue.__init__(self, None, bigdl_type, classes, iou, use_voc2007, skip_class)


class Loss(JavaValue):
    """
    This evaluation method is calculate loss of output with respect to target
    >>> from bigdl.dllib.nn.criterion import ClassNLLCriterion
    >>> loss = Loss()
    creating: createClassNLLCriterion
    creating: createLoss

    >>> loss = Loss(ClassNLLCriterion())
    creating: createClassNLLCriterion
    creating: createLoss
    """

    def __init__(self, cri=None, bigdl_type="float"):
        from bigdl.dllib.nn.criterion import ClassNLLCriterion
        if cri is None:
            cri = ClassNLLCriterion()
        JavaValue.__init__(self, None, bigdl_type, cri)


class HitRatio(JavaValue):
    """
    Hit Ratio(HR) used in recommandation application.
    HR intuitively measures whether the test item is present on the top-k list.

    >>> hr10 = HitRatio(k = 10)
    creating: createHitRatio
    """

    def __init__(self, k=10, neg_num=100, bigdl_type="float"):
        """
        Create hit ratio validation method.

        :param k: top k
        :param neg_num: number of negative items.
        """
        JavaValue.__init__(self, None, bigdl_type, k, neg_num)


class NDCG(JavaValue):
    """
    Normalized Discounted Cumulative Gain(NDCG).
    NDCG accounts for the position of the hit by assigning higher scores to hits at top ranks.

    >>> ndcg = NDCG(k = 10)
    creating: createNDCG
    """

    def __init__(self, k=10, neg_num=100, bigdl_type="float"):
        """
        Create NDCG validation method.

        :param k: top k
        :param neg_num: number of negative items.
        """
        JavaValue.__init__(self, None, bigdl_type, k, neg_num)


class MAE(JavaValue):
    """
    This evaluation method calculates the mean absolute error of output with respect to target.

    >>> mae = MAE()
    creating: createMAE
    """

    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type)


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


        :param interval: interval is the "n" where an action is triggeredevery "n" iterations
        """
        JavaValue.__init__(self, None, bigdl_type, interval)


class MaxScore(JavaValue):
    """
    A trigger that triggers an action when validation score larger than "max" score


    >>> maxScore = MaxScore(0.4)
    creating: createMaxScore
    """

    def __init__(self, max, bigdl_type="float"):
        """
        Create a MaxScore trigger.


        :param max: max score
        """
        JavaValue.__init__(self, None, bigdl_type, max)


class MinLoss(JavaValue):
    """
    A trigger that triggers an action when training loss less than "min" loss


    >>> minLoss = MinLoss(0.1)
    creating: createMinLoss
    """

    def __init__(self, min, bigdl_type="float"):
        """
        Create a MinLoss trigger.


        :param min: min loss
        """
        JavaValue.__init__(self, None, bigdl_type, min)


class TriggerAnd(JavaValue):
    """
    A trigger contains other triggers and triggers when all of them trigger (logical AND)


    >>> a = TriggerAnd(MinLoss(0.1), MaxEpoch(2))
    creating: createMinLoss
    creating: createMaxEpoch
    creating: createTriggerAnd
    """

    def __init__(self, first, *other):
        """
        Create a And trigger.


        :param first: first Trigger
        :param other: other Trigger
        """
        JavaValue.__init__(self, None, "float", first, list(other))


class TriggerOr(JavaValue):
    """
    A trigger contains other triggers and triggers when any of them trigger (logical OR)


    >>> o = TriggerOr(MinLoss(0.1), MaxEpoch(2))
    creating: createMinLoss
    creating: createMaxEpoch
    creating: createTriggerOr
    """

    def __init__(self, first, *other):
        """
        Create a Or trigger.


        :param first: first Trigger
        :param other: other Trigger
        """
        JavaValue.__init__(self, None, "float", first, list(other))


class Poly(JavaValue):
    """
    A learning rate decay policy, where the effective learning rate
    follows a polynomial decay, to be zero by the max_iteration.
    Calculation: base_lr (1 - iter/max_iteration) ^ (power)


    :param power: coeffient of decay, refer to calculation formula
    :param max_iteration: max iteration when lr becomes zero

    >>> poly = Poly(0.5, 2)
    creating: createPoly
    """

    def __init__(self, power, max_iteration, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, power, max_iteration)


class Exponential(JavaValue):
    """
    [[Exponential]] is a learning rate schedule, which rescale the learning rate by
    lr_{n + 1} = lr * decayRate `^` (iter / decayStep)
    :param decay_step the inteval for lr decay
    :param decay_rate decay rate
    :param stair_case if true, iter / decayStep is an integer division
                     and the decayed learning rate follows a staircase function.

    >>> exponential = Exponential(100, 0.1)
    creating: createExponential
    """

    def __init__(self, decay_step, decay_rate, stair_case=False, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, decay_step, decay_rate, stair_case)


class Step(JavaValue):
    """
    A learning rate decay policy, where the effective learning rate is
    calculated as base_lr * gamma ^ (floor(iter / step_size))


    :param step_size:
    :param gamma:


    >>> step = Step(2, 0.3)
    creating: createStep
    """

    def __init__(self, step_size, gamma, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, step_size, gamma)


class Default(JavaValue):
    """
    A learning rate decay policy, where the effective learning rate is
    calculated as base_lr * gamma ^ (floor(iter / step_size))

    :param step_size
    :param gamma

    >>> step = Default()
    creating: createDefault
    """

    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type)


class Plateau(JavaValue):
    """
    Plateau is the learning rate schedule when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor of 2-10
    once learning stagnates. It monitors a quantity and if no improvement
    is seen for a 'patience' number of epochs, the learning rate is reduced.

    :param monitor quantity to be monitored, can be Loss or score
    :param factor factor by which the learning rate will be reduced. new_lr = lr * factor
    :param patience number of epochs with no improvement after which learning rate will be reduced.
    :param mode one of {min, max}.
                In min mode, lr will be reduced when the quantity monitored has stopped decreasing;
                in max mode it will be reduced when the quantity monitored has stopped increasing
    :param epsilon threshold for measuring the new optimum, to only focus on significant changes.
    :param cooldown number of epochs to wait before resuming normal operation
                    after lr has been reduced.
    :param min_lr lower bound on the learning rate.

    >>> plateau = Plateau("score")
    creating: createPlateau
    """

    def __init__(self,
                 monitor,
                 factor=0.1,
                 patience=10,
                 mode="min",
                 epsilon=1e-4,
                 cooldown=0,
                 min_lr=0.0,
                 bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, monitor, factor, patience, mode, epsilon,
                           cooldown, min_lr)


class Warmup(JavaValue):
    """
    A learning rate gradual increase policy, where the effective learning rate
    increase delta after each iteration.
    Calculation: base_lr + delta * iteration

    :param delta: increase amount after each iteration

    >>> warmup = Warmup(0.05)
    creating: createWarmup
    """

    def __init__(self, delta, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, delta)


class SequentialSchedule(JavaValue):
    """
    Stack several learning rate schedulers.

    :param iterationPerEpoch: iteration numbers per epoch

    >>> sequentialSchedule = SequentialSchedule(5)
    creating: createSequentialSchedule
    >>> poly = Poly(0.5, 2)
    creating: createPoly
    >>> test = sequentialSchedule.add(poly, 5)



    """

    def __init__(self, iteration_per_epoch, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, iteration_per_epoch)

    def add(self, scheduler, max_iteration, bigdl_type="float"):
        """
        Add a learning rate scheduler to the contained `schedules`

        :param scheduler: learning rate scheduler to be add
        :param max_iteration: iteration numbers this scheduler will run
        """
        return callBigDlFunc(bigdl_type, "addScheduler", self.value, scheduler, max_iteration)


class OptimMethod(JavaValue):

    def __init__(self, jvalue, bigdl_type, *args):
        if (jvalue):
            invalidInputError((type(jvalue) == JavaObject),
                              f"jvalue type ${type(jvalue)} doesn't match"
                              f" JavaObject ${JavaObject}")
            self.value = jvalue
        else:
            self.value = callBigDlFunc(
                bigdl_type, JavaValue.jvm_class_constructor(self), *args)
        self.bigdl_type = bigdl_type

    @staticmethod
    def load(path, bigdl_type="float"):
        """
        load optim method
        :param path: file path
        """
        return callBigDlFunc(bigdl_type, "loadOptimMethod", path)

    def save(self, path, overWrite):
        """
        save OptimMethod
        :param path      path
        :param overWrite whether to overwrite
        """
        method = self.value
        return callBigDlFunc(self.bigdl_type, "saveOptimMethod", method, path, overWrite)


class SGD(OptimMethod):
    """
    A plain implementation of SGD

    :param learningrate learning rate
    :param learningrate_decay learning rate decay
    :param weightdecay weight decay
    :param momentum momentum
    :param dampening dampening for momentum
    :param nesterov enables Nesterov momentum
    :param learningrates 1D tensor of individual learning rates
    :param weightdecays 1D tensor of individual weight decays
    >>> sgd = SGD()
    creating: createDefault
    creating: createSGD
    """

    def __init__(self,
                 learningrate=1e-3,
                 learningrate_decay=0.0,
                 weightdecay=0.0,
                 momentum=0.0,
                 dampening=DOUBLEMAX,
                 nesterov=False,
                 leaningrate_schedule=None,
                 learningrates=None,
                 weightdecays=None,
                 bigdl_type="float"):
        super(SGD, self).__init__(None, bigdl_type, learningrate, learningrate_decay, weightdecay,
                                  momentum, dampening, nesterov,
                                  leaningrate_schedule if (leaningrate_schedule) else Default(),
                                  JTensor.from_ndarray(learningrates),
                                  JTensor.from_ndarray(weightdecays))


class Adagrad(OptimMethod):
    """
    An implementation of Adagrad. See the original paper:
    http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

    :param learningrate learning rate
    :param learningrate_decay learning rate decay
    :param weightdecay weight decay
    >>> adagrad = Adagrad()
    creating: createAdagrad
    """

    def __init__(self,
                 learningrate=1e-3,
                 learningrate_decay=0.0,
                 weightdecay=0.0,
                 bigdl_type="float"):
        super(Adagrad, self).__init__(None, bigdl_type, learningrate, learningrate_decay,
                                      weightdecay)


class LBFGS(OptimMethod):
    """
    This implementation of L-BFGS relies on a user-provided line
    search function (state.lineSearch). If this function is not
    provided, then a simple learningRate is used to produce fixed
    size steps. Fixed size steps are much less costly than line
    searches, and can be useful for stochastic problems.
    The learning rate is used even when a line search is provided.
    This is also useful for large-scale stochastic problems, where
    opfunc is a noisy approximation of f(x). In that case, the learning
    rate allows a reduction of confidence in the step size.

    :param max_iter Maximum number of iterations allowed
    :param max_eval Maximum number of function evaluations
    :param tolfun Termination tolerance on the first-order optimality
    :param tolx Termination tol on progress in terms of func/param changes
    :param ncorrection
    :param learningrate
    :param verbose
    :param linesearch A line search function
    :param linesearch_options If no line search provided, then a fixed step size is used
    >>> lbfgs = LBFGS()
    creating: createLBFGS
    """

    def __init__(self,
                 max_iter=20,
                 max_eval=DOUBLEMAX,
                 tolfun=1e-5,
                 tolx=1e-9,
                 ncorrection=100,
                 learningrate=1.0,
                 verbose=False,
                 linesearch=None,
                 linesearch_options=None,
                 bigdl_type="float"):
        if linesearch or linesearch_options:
            invalidInputError(False, 'linesearch and linesearch_options must be None in LBFGS')
        super(LBFGS, self).__init__(None, bigdl_type, max_iter, max_eval, tolfun, tolx,
                                    ncorrection, learningrate, verbose, linesearch,
                                    linesearch_options)


class Adadelta(OptimMethod):
    """
    Adadelta implementation for SGD: http://arxiv.org/abs/1212.5701

    :param decayrate interpolation parameter rho
    :param epsilon for numerical stability
    >>> adagrad = Adadelta()
    creating: createAdadelta
    """

    def __init__(self,
                 decayrate=0.9,
                 epsilon=1e-10,
                 bigdl_type="float"):
        super(Adadelta, self).__init__(None, bigdl_type, decayrate, epsilon)


class Adam(OptimMethod):
    """
    An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf
    :param learningrate learning rate
    :param learningrate_decay learning rate decay
    :param beta1 first moment coefficient
    :param beta2 second moment coefficient
    :param epsilon for numerical stability
    >>> adam = Adam()
    creating: createAdam
    """

    def __init__(self,
                 learningrate=1e-3,
                 learningrate_decay=0.0,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 bigdl_type="float"):
        super(Adam, self).__init__(None, bigdl_type, learningrate, learningrate_decay,
                                   beta1, beta2, epsilon)


class ParallelAdam(OptimMethod):
    """
    An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf
    :param learningrate learning rate
    :param learningrate_decay learning rate decay
    :param beta1 first moment coefficient
    :param beta2 second moment coefficient
    :param epsilon for numerical stability
    >>> init_engine()
    >>> pAdam = ParallelAdam()
    creating: createParallelAdam
    """

    def __init__(self,
                 learningrate=1e-3,
                 learningrate_decay=0.0,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 parallel_num=-1,
                 bigdl_type="float"):
        if parallel_num == -1:
            parallel_num = get_node_and_core_number()[1]
        super(ParallelAdam, self).__init__(None, bigdl_type, learningrate, learningrate_decay,
                                           beta1, beta2, epsilon, parallel_num)


class Ftrl(OptimMethod):
    """
    An implementation of Ftrl https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf.
    Support L1 penalty, L2 penalty and shrinkage-type L2 penalty.

    :param learningrate learning rate
    :param learningrate_power double, must be less or equal to zero. Default is -0.5.
    :param initial_accumulator_value double, the starting value for accumulators,
        require zero or positive values.
    :param l1_regularization_strength double, must be greater or equal to zero. Default is zero.
    :param l2_regularization_strength double, must be greater or equal to zero. Default is zero.
    :param l2_shrinkage_regularization_strength double, must be greater or equal to zero.
        Default is zero. This differs from l2RegularizationStrength above. L2 above is a
        stabilization penalty, whereas this one is a magnitude penalty.
    >>> ftrl = Ftrl()
    creating: createFtrl
    >>> ftrl2 = Ftrl(1e-2, -0.1, 0.2, 0.3, 0.4, 0.5)
    creating: createFtrl
    """

    def __init__(self,
                 learningrate=1e-3,
                 learningrate_power=-0.5,
                 initial_accumulator_value=0.1,
                 l1_regularization_strength=0.0,
                 l2_regularization_strength=0.0,
                 l2_shrinkage_regularization_strength=0.0,
                 bigdl_type="float"):
        super(Ftrl, self).__init__(None, bigdl_type, learningrate, learningrate_power,
                                   initial_accumulator_value,
                                   l1_regularization_strength,
                                   l2_regularization_strength,
                                   l2_shrinkage_regularization_strength)


class Adamax(OptimMethod):
    """
    An implementation of Adamax http://arxiv.org/pdf/1412.6980.pdf
    :param learningrate learning rate
    :param beta1 first moment coefficient
    :param beta2 second moment coefficient
    :param epsilon for numerical stability
    >>> adagrad = Adamax()
    creating: createAdamax
    """

    def __init__(self,
                 learningrate=0.002,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-38,
                 bigdl_type="float"):
        super(Adamax, self).__init__(None, bigdl_type, learningrate, beta1, beta2, epsilon)


class RMSprop(OptimMethod):
    """
    An implementation of RMSprop
    :param learningrate learning rate
    :param learningrate_decay learning rate decay
    :param decayrate decay rate, also called rho
    :param epsilon for numerical stability
    >>> adagrad = RMSprop()
    creating: createRMSprop
    """

    def __init__(self,
                 learningrate=1e-2,
                 learningrate_decay=0.0,
                 decayrate=0.99,
                 epsilon=1e-8,
                 bigdl_type="float"):
        super(RMSprop, self).__init__(None, bigdl_type, learningrate, learningrate_decay, decayrate,
                                      epsilon)


class MultiStep(JavaValue):
    """
    similar to step but it allows non uniform steps defined by stepSizes


    :param step_size: the series of step sizes used for lr decay
    :param gamma: coefficient of decay


    >>> step = MultiStep([2, 5], 0.3)
    creating: createMultiStep
    """

    def __init__(self, step_sizes, gamma, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, step_sizes, gamma)


class BaseOptimizer(JavaValue):

    def set_model(self, model):
        """
        Set model.


        :param model: new model
        """
        self.value.setModel(model.value)

    def set_criterion(self, criterion):
        """
        set new criterion, for optimizer reuse

        :param criterion: new criterion
        :return:
        """
        callBigDlFunc(self.bigdl_type, "setCriterion", self.value,
                      criterion)

    def set_checkpoint(self, checkpoint_trigger,
                       checkpoint_path, isOverWrite=True):
        """
        Configure checkpoint settings.


        :param checkpoint_trigger: the interval to write snapshots
        :param checkpoint_path: the path to write snapshots into
        :param isOverWrite: whether to overwrite existing snapshots in path.default is True
        """
        if not os.path.exists(checkpoint_path):
            mkpath(checkpoint_path)
        callBigDlFunc(self.bigdl_type, "setCheckPoint", self.value,
                      checkpoint_trigger, checkpoint_path, isOverWrite)

    def set_gradclip_const(self, min_value, max_value):
        """
        Configure constant clipping settings.


        :param min_value: the minimum value to clip by
        :param max_value: the maxmimum value to clip by
        """
        callBigDlFunc(self.bigdl_type, "setConstantClip", self.value, min_value, max_value)

    def set_gradclip_l2norm(self, clip_norm):
        """
        Configure L2 norm clipping settings.


        :param clip_norm: gradient L2-Norm threshold
        """
        callBigDlFunc(self.bigdl_type, "setL2NormClip", self.value, clip_norm)

    def disable_gradclip(self):
        """
        disable clipping.
        """
        callBigDlFunc(self.bigdl_type, "disableClip", self.value)

    # return a module
    def optimize(self):
        """
        Do an optimization.
        """
        jmodel = callJavaFunc(self.value.optimize)
        from bigdl.dllib.nn.layer import Layer
        return Layer.of(jmodel)

    def set_train_summary(self, summary):
        """
        Set train summary. A TrainSummary object contains information
        necessary for the optimizer to know how often the logs are recorded,
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
        necessary for the optimizer to know how often the logs are recorded,
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

    def set_end_when(self, end_when):
        """
        When to stop, passed in a [[Trigger]]
        """
        self.value.setEndWhen(end_when.value)
        return self


class Optimizer(BaseOptimizer):

    # NOTE: This is a deprecated method, you should use `create` method instead.
    def __init__(self,
                 model,
                 training_rdd,
                 criterion,
                 end_trigger,
                 batch_size,
                 optim_method=None,
                 bigdl_type="float"):
        """
        Create a distributed optimizer.


        :param model: the neural net model
        :param training_rdd: the training dataset
        :param criterion: the loss function
        :param optim_method: the algorithm to use for optimization,
           e.g. SGD, Adagrad, etc. If optim_method is None, the default algorithm is SGD.
        :param end_trigger: when to end the optimization
        :param batch_size: training batch size
        """
        warnings.warn("You are recommended to use `create` method to create an optimizer.")
        invalidInputError(isinstance(training_rdd, RDD), "Only type of RDD is allowed")
        self.pvalue = DistriOptimizer(model,
                                      training_rdd,
                                      criterion,
                                      end_trigger,
                                      batch_size,
                                      optim_method,
                                      bigdl_type)
        self.value = self.pvalue.value
        self.bigdl_type = self.pvalue.bigdl_type

    @staticmethod
    def create(model,
               training_set,
               criterion,
               end_trigger=None,
               batch_size=32,
               optim_method=None,
               cores=None,
               bigdl_type="float"):
        """
        Create an optimizer.
        Depend on the input type, the returning optimizer can be a local optimizer \
        or a distributed optimizer.

        :param model: the neural net model
        :param training_set: (features, label) for local mode. RDD[Sample] for distributed mode.
        :param criterion: the loss function
        :param optim_method: the algorithm to use for optimization,
           e.g. SGD, Adagrad, etc. If optim_method is None, the default algorithm is SGD.
        :param end_trigger: when to end the optimization. default value is MapEpoch(1)
        :param batch_size: training batch size
        :param cores: This is for local optimizer only and use total physical cores as the default
        value
        """
        if not end_trigger:
            end_trigger = MaxEpoch(1)
        if not optim_method:
            optim_method = SGD()
        if isinstance(training_set, RDD) or isinstance(training_set, DataSet):
            return DistriOptimizer(model=model,
                                   training_rdd=training_set,
                                   criterion=criterion,
                                   end_trigger=end_trigger,
                                   batch_size=batch_size,
                                   optim_method=optim_method,
                                   bigdl_type=bigdl_type)
        elif isinstance(training_set, tuple) and len(training_set) == 2:
            x, y = training_set
            return LocalOptimizer(X=x,
                                  Y=y,
                                  model=model,
                                  criterion=criterion,
                                  end_trigger=end_trigger,
                                  batch_size=batch_size,
                                  optim_method=optim_method,
                                  cores=cores,
                                  bigdl_type="float")
        else:
            invalidInputError(False, "Not supported training set: %s" % type(training_set))

    def set_validation(self, batch_size, val_rdd, trigger, val_method=None):
        """
        Configure validation settings.


        :param batch_size: validation batch size
        :param val_rdd: validation dataset
        :param trigger: validation interval
        :param val_method: the ValidationMethod to use,e.g. "Top1Accuracy", "Top5Accuracy", "Loss"
        """
        if val_method is None:
            val_method = [Top1Accuracy()]
        func_name = "setValidation"
        if isinstance(val_rdd, DataSet):
            func_name = "setValidationFromDataSet"
        callBigDlFunc(self.bigdl_type, func_name, self.value, batch_size,
                      trigger, val_rdd, to_list(val_method))

    def set_traindata(self, training_rdd, batch_size):
        """
        Set new training dataset, for optimizer reuse

        :param training_rdd: the training dataset
        :param batch_size: training batch size
        :return:
        """
        callBigDlFunc(self.bigdl_type, "setTrainData", self.value,
                      training_rdd, batch_size)


class DistriOptimizer(Optimizer):
    def __init__(self,
                 model,
                 training_rdd,
                 criterion,
                 end_trigger,
                 batch_size,
                 optim_method=None,
                 bigdl_type="float"):
        """
        Create an optimizer.


        :param model: the neural net model
        :param training_data: the training dataset
        :param criterion: the loss function
        :param optim_method: the algorithm to use for optimization,
           e.g. SGD, Adagrad, etc. If optim_method is None, the default algorithm is SGD.
        :param end_trigger: when to end the optimization
        :param batch_size: training batch size
        """
        if not optim_method:
            optim_methods = {model.name(): SGD()}
        elif isinstance(optim_method, OptimMethod):
            optim_methods = {model.name(): optim_method}
        elif isinstance(optim_method, JavaObject):
            optim_methods = {model.name(): OptimMethod(optim_method, bigdl_type)}
        else:
            optim_methods = optim_method
        if isinstance(training_rdd, RDD):
            JavaValue.__init__(self, None, bigdl_type, model.value,
                               training_rdd, criterion,
                               optim_methods, end_trigger, batch_size)
        elif isinstance(training_rdd, DataSet):
            self.bigdl_type = bigdl_type
            self.value = callBigDlFunc(self.bigdl_type, "createDistriOptimizerFromDataSet",
                                       model.value, training_rdd, criterion,
                                       optim_methods, end_trigger, batch_size)


class LocalOptimizer(BaseOptimizer):
    """
    Create an optimizer.


    :param model: the neural net model
    :param X: the training features which is an ndarray or list of ndarray
    :param Y: the training label which is an ndarray
    :param criterion: the loss function
    :param optim_method: the algorithm to use for optimization,
       e.g. SGD, Adagrad, etc. If optim_method is None, the default algorithm is SGD.
    :param end_trigger: when to end the optimization
    :param batch_size: training batch size
    :param cores: by default is the total physical cores.
    """

    def __init__(self,
                 X,
                 Y,
                 model,
                 criterion,
                 end_trigger,
                 batch_size,
                 optim_method=None,
                 cores=None,
                 bigdl_type="float"):
        if not optim_method:
            optim_methods = {model.name(): SGD()}
        elif isinstance(optim_method, OptimMethod):
            optim_methods = {model.name(): optim_method}
        elif isinstance(optim_method, JavaObject):
            optim_methods = {model.name(): OptimMethod(optim_method, bigdl_type)}
        else:
            optim_methods = optim_method
        if cores is None:
            cores = multiprocessing.cpu_count()
        JavaValue.__init__(self, None, bigdl_type,
                           [JTensor.from_ndarray(X) for X in to_list(X)],
                           JTensor.from_ndarray(Y),
                           model.value,
                           criterion,
                           optim_methods, end_trigger, batch_size, cores)

    def set_validation(self, batch_size, X_val, Y_val, trigger, val_method=None):
        """
        Configure validation settings.

        :param batch_size: validation batch size
        :param X_val: features of validation dataset
        :param Y_val: label of validation dataset
        :param trigger: validation interval
        :param val_method: the ValidationMethod to use,e.g. "Top1Accuracy", "Top5Accuracy", "Loss"
        """
        if val_method is None:
            val_method = [Top1Accuracy()]
        callBigDlFunc(self.bigdl_type, "setValidation", self.value, batch_size,
                      trigger, [JTensor.from_ndarray(X) for X in to_list(X_val)],
                      JTensor.from_ndarray(Y_val), to_list(val_method))


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


        :param tag: the type of the logs, Supported tags are: "LearningRate","Loss", "Throughput"
        """
        return callBigDlFunc(self.bigdl_type, "summaryReadScalar", self.value,
                             tag)

    def set_summary_trigger(self, name, trigger):
        """
        Set the interval of recording for each indicator.


        :param tag: tag name. Supported tag names are "LearningRate", "Loss","Throughput",
        "Parameters". "Parameters" is an umbrella tag thatincludes weight, bias, gradWeight,
         gradBias, and some running status(eg. runningMean and runningVar in BatchNormalization).
         If youdidn't set any triggers, we will by default record Loss and Throughputin each
         iteration, while *NOT* recording LearningRate and Parameters,as recording parameters may
         introduce substantial overhead when themodel is very big, LearningRate is not a public
         attribute for allOptimMethod.
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


        :param tag: the type of the logs. The tag should match the name ofthe ValidationMethod set
         into the optimizer. e.g."Top1AccuracyLoss","Top1Accuracy" or "Top5Accuracy".
        """
        return callBigDlFunc(self.bigdl_type, "summaryReadScalar", self.value,
                             tag)


class L1L2Regularizer(JavaValue):
    """
    Apply both L1 and L2 regularization

    :param l1 l1 regularization rate
    :param l2 l2 regularization rate

    """

    def __init__(self, l1, l2, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, l1, l2)


class ActivityRegularization(JavaValue):
    """
    Apply both L1 and L2 regularization

    :param l1 l1 regularization rate
    :param l2 l2 regularization rate

    """

    def __init__(self, l1, l2, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, l1, l2)


class L1Regularizer(JavaValue):
    """
    Apply L1 regularization

    :param l1 l1 regularization rate

    """

    def __init__(self, l1, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, l1)


class L2Regularizer(JavaValue):
    """
    Apply L2 regularization

    :param l2 l2 regularization rate

    """

    def __init__(self, l2, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, l2)


def _test():
    import doctest
    from pyspark import SparkContext
    from bigdl.dllib.optim import optimizer
    from bigdl.dllib.utils.common import init_engine
    from bigdl.dllib.utils.common import create_spark_conf
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
