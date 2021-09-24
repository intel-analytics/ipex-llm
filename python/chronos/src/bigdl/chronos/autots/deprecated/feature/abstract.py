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


class BaseFeatureTransformer(ABC):
    """
    Abstract Base class for Feature transformers.
    """
    check_optional_config = False

    @abstractmethod
    def fit_transform(self, input_df, **config):
        """
        fit data with the input dataframe
        Will refit the scalars to this data if any.
        :param input_df: input to be fitted
        :param config: the config
        :return:
        """
        pass

    @abstractmethod
    def transform(self, input_df):
        """
        transform the data with fitted
        :param input_df: input dataframe
        :return:
        """
        pass

    @abstractmethod
    def save(self, file_path):
        """
        save the feature tools internal variables.
        Some of the variables are derived after fit_transform, so only saving config is not enough.
        :param: file_path : the file to be saved
        :param: config: the trial config
        :return:
        """
        pass

    @abstractmethod
    def restore(self, **config):
        """
        Restore variables from file
        :param file_path: file contain saved parameters.
                          i.e. some parameters are obtained during training,
                          not in trial config, e.g. scaler fit params)
        :param config: the trial config
        :return:
        """
        pass

    @abstractmethod
    def _get_required_parameters(self):
        """
        :return: required parameters to be set into config
        """
        return set()

    @abstractmethod
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
