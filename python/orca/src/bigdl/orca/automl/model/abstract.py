#
# Copyright 2018 Analytics Zoo Authors.
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


class BaseModel(ABC):
    """
    base model for automl tuning
    """

    check_optional_config = False
    config = None
    model = None

    @abstractmethod
    def fit_eval(self, data, validation_data=None, **kwargs):
        """
        optimize and evaluate for one iteration for tuning
        :param data: train data
        :param validation_data: validation data

        :return: A Dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, checkpoint):
        """
        Save the model at the provided checkpoint.
        :param checkpoint: (str) Path to the target checkpoint file.

        :return: checkpoint path
        """
        pass

    @abstractmethod
    def restore(self, checkpoint):
        """
        Restore the model from the provided checkpoint.
        :param checkpoint: (str) Path to target checkpoint file.

        :return:
        """
        pass

    def get_model(self):
        return self.model

    def _get_required_parameters(self):
        """
        :return: required parameters to be set into config
        """
        return set()

    def _get_optional_parameters(self):
        """
        :return: optional parameters to be set into config
        """
        return set()

    def _check_config(self, **config):
        """
        Do necessary checking for config
        :param config:
        :return:
        """
        config_parameters = set(config.keys())
        if not config_parameters.issuperset(self._get_required_parameters()):
            raise ValueError("Missing required parameters in configuration. " +
                             "Required parameters are: " + str(self._get_required_parameters()))
        if self.check_optional_config and \
                not config_parameters.issuperset(self._get_optional_parameters()):
            raise ValueError("Missing optional parameters in configuration. " +
                             "Optional parameters are: " + str(self._get_optional_parameters()))
        return True


class ModelBuilder:

    @abstractmethod
    def build(self, config):
        pass
