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


class Callback(object):
    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

    def before_train_epoch(self, runner):
        self.before_epoch(runner)

    def before_val_epoch(self, runner):
        self.before_epoch(runner)

    def after_train_epoch(self, runner):
        self.after_epoch(runner)

    def after_val_epoch(self, runner):
        self.after_epoch(runner)

    def before_train_iter(self, runner):
        self.before_iter(runner)

    def before_val_iter(self, runner):
        self.before_iter(runner)

    def after_train_iter(self, runner):
        self.after_iter(runner)

    def after_val_iter(self, runner):
        self.after_iter(runner)
