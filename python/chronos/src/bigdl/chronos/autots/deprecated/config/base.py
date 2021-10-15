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

from abc import ABCMeta, abstractmethod


class Recipe(metaclass=ABCMeta):
    """
    Recipe
    """

    def __init__(self):
        # ----- runtime parameters
        self.training_iteration = 1
        self.num_samples = 1
        self.reward_metric = None

    @abstractmethod
    def search_space(self):
        pass

    def runtime_params(self):
        runtime_config = {
            "training_iteration": self.training_iteration,
            "num_samples": self.num_samples,
        }
        if self.reward_metric is not None:
            runtime_config["reward_metric"] = self.reward_metric
        return runtime_config
