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
from abc import ABC, abstractmethod

from bigdl.dllib.utils.common import DOUBLEMAX
from bigdl.orca.learn.optimizers.schedule import Scheduler
from bigdl.dllib.utils.log4Error import *


class Optimizer(ABC):

    @abstractmethod
    def get_optimizer(self):
        pass


class SGD(Optimizer):
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
                 learningrate_schedule=None,
                 learningrates=None,
                 weightdecays=None):
        from bigdl.dllib.optim.optimizer import SGD as BSGD
        invalidInputError(isinstance(learningrate_schedule, Scheduler),
                          "learningrate_schedule should be an "
                          "bigdl.orca.learn.optimizers.schedule.Scheduler,"
                          " but got {learningrate_schedule}")
        self.optimizer = BSGD(learningrate,
                              learningrate_decay,
                              weightdecay,
                              momentum,
                              dampening,
                              nesterov,
                              learningrate_schedule.get_scheduler(),
                              learningrates,
                              weightdecays,
                              bigdl_type="float")

    def get_optimizer(self):
        return self.optimizer


class Adagrad(Optimizer):
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
                 weightdecay=0.0):
        from bigdl.dllib.optim.optimizer import Adagrad as BAdagrad
        self.optimizer = BAdagrad(learningrate, learningrate_decay,
                                  weightdecay, bigdl_type="float")

    def get_optimizer(self):
        return self.optimizer


class LBFGS(Optimizer):
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
                 linesearch_options=None):
        from bigdl.dllib.optim.optimizer import LBFGS as BLBFGS
        self.optimizer = BLBFGS(
            max_iter,
            max_eval,
            tolfun,
            tolx,
            ncorrection,
            learningrate,
            verbose,
            linesearch,
            linesearch_options,
            bigdl_type="float"
        )

    def get_optimizer(self):
        return self.optimizer


class Adadelta(Optimizer):
    """
    Adadelta implementation for SGD: http://arxiv.org/abs/1212.5701

    :param decayrate interpolation parameter rho
    :param epsilon for numerical stability
    >>> adagrad = Adadelta()
    creating: createAdadelta
    """

    def __init__(self,
                 decayrate=0.9,
                 epsilon=1e-10):
        from bigdl.dllib.optim.optimizer import Adadelta as BAdadelta
        self.optimizer = BAdadelta(decayrate,
                                   epsilon,
                                   bigdl_type="float")

    def get_optimizer(self):
        return self.optimizer


class Adam(Optimizer):
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
                 epsilon=1e-8):
        from bigdl.dllib.optim.optimizer import Adam as BAdam
        self.optimizer = BAdam(learningrate,
                               learningrate_decay,
                               beta1,
                               beta2,
                               epsilon,
                               bigdl_type="float")

    def get_optimizer(self):
        return self.optimizer


class ParallelAdam(Optimizer):
    """
    An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf
    :param learningrate learning rate
    :param learningrate_decay learning rate decay
    :param beta1 first moment coefficient
    :param beta2 second moment coefficient
    :param epsilon for numerical stability
    >>> pAdam = ParallelAdam()
    creating: createParallelAdam
    """

    def __init__(self,
                 learningrate=1e-3,
                 learningrate_decay=0.0,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 parallel_num=-1):
        from bigdl.dllib.optim.optimizer import ParallelAdam as BParallelAdam
        self.optimizer = BParallelAdam(learningrate,
                                       learningrate_decay,
                                       beta1,
                                       beta2,
                                       epsilon,
                                       parallel_num,
                                       bigdl_type="float")

    def get_optimizer(self):
        return self.optimizer


class Ftrl(Optimizer):
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
                 l2_shrinkage_regularization_strength=0.0):
        from bigdl.dllib.optim.optimizer import Ftrl as BFtrl
        self.optimizer = BFtrl(learningrate,
                               learningrate_power,
                               initial_accumulator_value,
                               l1_regularization_strength,
                               l2_regularization_strength,
                               l2_shrinkage_regularization_strength,
                               bigdl_type="float")

    def get_optimizer(self):
        return self.optimizer


class Adamax(Optimizer):
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
                 epsilon=1e-38):
        from bigdl.dllib.optim.optimizer import Adamax as BAdamax
        self.optimizer = BAdamax(learningrate,
                                 beta1,
                                 beta2,
                                 epsilon,
                                 bigdl_type="float")

    def get_optimizer(self):
        return self.optimizer


class RMSprop(Optimizer):
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
                 epsilon=1e-8):
        from bigdl.dllib.optim.optimizer import RMSprop as BRMSprop
        self.optimizer = BRMSprop(learningrate,
                                  learningrate_decay,
                                  decayrate,
                                  epsilon,
                                  bigdl_type="float")

    def get_optimizer(self):
        return self.optimizer
