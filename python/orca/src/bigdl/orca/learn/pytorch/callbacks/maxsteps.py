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
from bigdl.orca.learn.pytorch.callbacks import Callback
import math


class MaxstepsCallback(Callback):

    def __init__(self, max_step) -> None:
        super().__init__()
        self.max_step = max_step

    def before_run(self, runner):
        runner.num_epochs = math.ceil(self.max_step / len(runner.train_loader))

    def after_train_iter(self, runner):
        if runner.global_step >= self.max_step:
            runner.stop = True
