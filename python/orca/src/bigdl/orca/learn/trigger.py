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


class Trigger(ABC):
    @abstractmethod
    def get_trigger(self):
        pass

    @staticmethod
    def convert_trigger(trigger):
        if trigger is None:
            return None
        if isinstance(trigger, str):
            if trigger.lower() == "everyepoch":
                return EveryEpoch().get_trigger()
            else:
                raise ValueError("Only 'EveryEpoch', orca triggers and bigdl triggers are "
                                 "supported now")
        elif isinstance(trigger, Trigger):
            return trigger.get_trigger()
        else:
            return trigger


class EveryEpoch(Trigger):
    """
    A trigger specifies a timespot or several timespots during training,
    and a corresponding action will be taken when the timespot(s) is reached.
    EveryEpoch is a trigger that triggers an action when each epoch finishs.
    Could be used as trigger in setvalidation and setcheckpoint in Optimizer,
    and also in TrainSummary.set_summary_trigger.


    >>> everyEpoch = EveryEpoch()
    """
    def __init__(self):
        from bigdl.dllib.optim.optimizer import EveryEpoch
        self.trigger = EveryEpoch()

    def get_trigger(self):
        return self.trigger


class SeveralIteration(Trigger):
    """
    A trigger specifies a timespot or several timespots during training,
    and a corresponding action will be taken when the timespot(s) is reached.
    SeveralIteration is a trigger that triggers an action every "n"
    iterations.
    Could be used as trigger in setvalidation and setcheckpoint in Optimizer,
    and also in TrainSummary.set_summary_trigger.


    >>> serveralIteration = SeveralIteration(2)
    """
    def __init__(self, interval):
        from bigdl.dllib.optim.optimizer import SeveralIteration
        self.trigger = SeveralIteration(interval)

    def get_trigger(self):
        return self.trigger
