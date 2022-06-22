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
    def __init__(self):
        self.model = None
        self.params = None
        self.trainer = None

    @abstractmethod
    def on_batch_begin(self, batch):
        """
        Called at the beginning of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        @param batch: Integer, index of batch within the current epoch.
        """
        pass

    @abstractmethod
    def on_batch_end(self, batch):
        """
        Called at the end of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        @param batch: Integer, index of batch within the current epoch.
        :param logs: Dict. Aggregated metric results up until this batch.
        """
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch):
        """
        Called at the start of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        @param epoch: Integer, index of epoch.
        @param logs: Dict. Currently, saved stats in last epoch has been passed to this argument
        for this method but may change in the future.
        """
        pass

    @abstractmethod
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        @param epoch:  Integer, index of epoch.
        @param logs: Dict, metric results for this training epoch, and for the validation epoch if
            validation is performed. Validation result keys are prefixed with val_. For training
            epoch, the values of the Model's metrics are returned.
            Example : {'loss': 0.2, 'accuracy': 0.7}
        """
        pass

    @abstractmethod
    def on_train_begin(self):
        """
        Called at the beginning of training.
        Subclasses should override for any actions to run.
        @param logs: Dict. Currently, no data is passed to this argument for this method
          but that may change in the future.
        """
        pass

    @abstractmethod
    def on_train_end(self, logs=None):
        """
        Called at the end of training.
        Subclasses should override for any actions to run.
        :param logs: Dict. Currently the output of the last call to on_epoch_end() is passed to
            this argument for this method but that may change in the future.
        """
        pass

    def set_model(self, model):
        self.model = model

    def set_param(self, param):
        self.params = param
