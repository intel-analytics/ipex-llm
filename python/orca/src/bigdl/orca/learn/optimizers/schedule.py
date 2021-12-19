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


class Scheduler(ABC):

    @abstractmethod
    def get_scheduler(self):
        pass


class Poly(Scheduler):
    """
    A learning rate decay policy, where the effective learning rate
    follows a polynomial decay, to be zero by the max_iteration.
    Calculation: base_lr (1 - iter/max_iteration) ^ (power)


    :param power: coeffient of decay, refer to calculation formula
    :param max_iteration: max iteration when lr becomes zero

    >>> poly = Poly(0.5, 2)
    creating: createPoly
    """
    def __init__(self, power, max_iteration):
        from bigdl.dllib.optim.optimizer import Poly as BPoly
        self.scheduler = BPoly(power, max_iteration)

    def get_scheduler(self):
        return self.scheduler


class Exponential(Scheduler):
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
    def __init__(self, decay_step, decay_rate, stair_case=False):
        from bigdl.dllib.optim.optimizer import Exponential as BExponential
        self.scheduler = BExponential(decay_step, decay_rate, stair_case)

    def get_scheduler(self):
        return self.scheduler


class Step(Scheduler):
    """
    A learning rate decay policy, where the effective learning rate is
    calculated as base_lr * gamma ^ (floor(iter / step_size))


    :param step_size:
    :param gamma:


    >>> step = Step(2, 0.3)
    creating: createStep
    """
    def __init__(self, step_size, gamma):
        from bigdl.dllib.optim.optimizer import Step as BStep

        self.scheduler = BStep(step_size, gamma)

    def get_scheduler(self):
        return self.scheduler


class Default(Scheduler):
    """
    A learning rate decay policy, where the effective learning rate is
    calculated as base_lr * gamma ^ (floor(iter / step_size))

    :param step_size
    :param gamma

    >>> step = Default()
    creating: createDefault
    """
    def __init__(self):
        from bigdl.dllib.optim.optimizer import Default as BDefault

        self.scheduler = BDefault()

    def get_scheduler(self):
        return self.scheduler


class Plateau(Scheduler):
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
                 min_lr=0.0):
        from bigdl.dllib.optim.optimizer import Plateau as BPlateau

        self.scheduler = BPlateau(monitor, factor, patience, mode, epsilon,
                                  cooldown, min_lr)

    def get_scheduler(self):
        return self.scheduler


class Warmup(Scheduler):
    """
    A learning rate gradual increase policy, where the effective learning rate
    increase delta after each iteration.
    Calculation: base_lr + delta * iteration

    :param delta: increase amount after each iteration

    >>> warmup = Warmup(0.05)
    creating: createWarmup
    """
    def __init__(self, delta):
        from bigdl.dllib.optim.optimizer import Warmup as BWarmup

        self.scheduler = BWarmup(delta)

    def get_scheduler(self):
        return self.scheduler


class MultiStep(Scheduler):
    """
    similar to step but it allows non uniform steps defined by stepSizes


    :param step_size: the series of step sizes used for lr decay
    :param gamma: coefficient of decay


    >>> step = MultiStep([2, 5], 0.3)
    creating: createMultiStep
    """
    def __init__(self, step_sizes, gamma):
        from bigdl.dllib.optim.optimizer import MultiStep as BMultiStep

        self.scheduler = BMultiStep(step_sizes, gamma)

    def get_scheduler(self):
        return self.scheduler


class SequentialSchedule(Scheduler):
    """
    Stack several learning rate schedulers.

    :param iterationPerEpoch: iteration numbers per epoch

    >>> sequentialSchedule = SequentialSchedule(5)
    creating: createSequentialSchedule
    >>> poly = Poly(0.5, 2)
    creating: createPoly
    >>> test = sequentialSchedule.add(poly, 5)



    """
    def __init__(self, iteration_per_epoch):
        from bigdl.dllib.optim.optimizer import SequentialSchedule as BSequentialSchedule

        self.scheduler = BSequentialSchedule(iteration_per_epoch)

    def get_scheduler(self):
        return self.scheduler

    def add(self, scheduler, max_iteration):
        """
        Add a learning rate scheduler to the contained `schedules`

        :param scheduler: learning rate scheduler to be add
        :param max_iteration: iteration numbers this scheduler will run
        """
        return self.get_scheduler().add(scheduler.get_scheduler(), max_iteration)
