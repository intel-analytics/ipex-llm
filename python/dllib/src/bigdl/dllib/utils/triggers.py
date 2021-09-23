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
from abc import abstractmethod
from bigdl.dllib.utils.common import JavaValue


class ZooTrigger(JavaValue):
    def jvm_class_constructor(self):
        name = "createZoo" + self.__class__.__name__
        print("creating: " + name)
        return name


class EveryEpoch(ZooTrigger):
    """
    A trigger specifies a timespot or several timespots during training,
    and a corresponding action will be taken when the timespot(s) is reached.
    EveryEpoch is a trigger that triggers an action when each epoch finishs.
    Could be used as trigger in setvalidation and setcheckpoint in Optimizer,
    and also in TrainSummary.set_summary_trigger.

    >>> everyEpoch = EveryEpoch()
    creating: createZooEveryEpoch
    """
    def __init__(self):
        """
        Create a EveryEpoch trigger.
        """
        ZooTrigger.__init__(self, None, "float")


class SeveralIteration(ZooTrigger):
    """
    A trigger specifies a timespot or several timespots during training,
    and a corresponding action will be taken when the timespot(s) is reached.
    SeveralIteration is a trigger that triggers an action every "n"
    iterations.
    Could be used as trigger in setvalidation and setcheckpoint in Optimizer,
    and also in TrainSummary.set_summary_trigger.

    >>> serveralIteration = SeveralIteration(2)
    creating: createZooSeveralIteration
    """
    def __init__(self, interval):
        """
        Create a SeveralIteration trigger.


        :param interval: interval is the "n" where an action is triggeredevery "n" iterations.

        """
        ZooTrigger.__init__(self, None, "float", interval)


class MaxEpoch(ZooTrigger):
    """
    A trigger specifies a timespot or several timespots during training,
    and a corresponding action will be taken when the timespot(s) is reached.
    MaxEpoch is a trigger that triggers an action when training reaches
    the number of epochs specified by "max_epoch".
    Usually used as end_trigger when creating an Optimizer.


    >>> maxEpoch = MaxEpoch(2)
    creating: createZooMaxEpoch
    """
    def __init__(self, max):
        """
        Create a MaxEpoch trigger.

        :param max_epoch: max_epoch
        """
        ZooTrigger.__init__(self, None, "float", max)


class MaxIteration(ZooTrigger):
    """
    A trigger specifies a timespot or several timespots during training,
    and a corresponding action will be taken when the timespot(s) is reached.
    MaxIteration is a trigger that triggers an action when training reaches
    the number of iterations specified by "max".
    Usually used as end_trigger when creating an Optimizer.


    >>> maxIteration = MaxIteration(2)
    creating: createZooMaxIteration
    """
    def __init__(self, max):
        """
        Create a MaxIteration trigger.


        :param max: max
        """
        ZooTrigger.__init__(self, None, "float", max)


class MaxScore(ZooTrigger):
    """
    A trigger that triggers an action when validation score larger than "max" score.


    >>> maxScore = MaxScore(0.7)
    creating: createZooMaxScore
    """
    def __init__(self, max):
        """
        Create a MaxScore trigger.


        :param max: max score
        """
        ZooTrigger.__init__(self, None, "float", max)


class MinLoss(ZooTrigger):
    """
     A trigger that triggers an action when training loss less than "min" loss.


    >>> minLoss = MinLoss(0.1)
    creating: createZooMinLoss
    """
    def __init__(self, min):
        """
        Create a MinLoss trigger.


        :param min: min loss
        """
        ZooTrigger.__init__(self, None, "float", min)


class TriggerAnd(ZooTrigger):
    """
    A trigger contains other triggers and triggers when all of them trigger (logical AND)


    >>> a = TriggerAnd(MinLoss(0.1), MaxEpoch(2))
    creating: createZooMinLoss
    creating: createZooMaxEpoch
    creating: createZooTriggerAnd
    """
    def __init__(self, first, *other):
        """
        Create a And trigger.


        :param first: first ZooTrigger
        :param other: other ZooTrigger
        """
        ZooTrigger.__init__(self, None, "float", first, list(other))


class TriggerOr(ZooTrigger):
    """
    A trigger contains other triggers and triggers when any of them trigger (logical OR)


    >>> o = TriggerOr(MinLoss(0.1), MaxEpoch(2))
    creating: createZooMinLoss
    creating: createZooMaxEpoch
    creating: createZooTriggerOr
    """
    def __init__(self, first, *other):
        """
        Create a Or trigger.


        :param first: first ZooTrigger
        :param other: other ZooTrigger
        """
        ZooTrigger.__init__(self, None, "float", first, list(other))
