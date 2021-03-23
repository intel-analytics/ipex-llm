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


class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, data, epochs, batch_size):
        """
        Train the model with train data.

        :param data: train data.
        :param epochs: number of epochs to train.
        :param batch_size: total batch size for each iteration.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data, batch_size):
        """
        Predict input data

        :param data: data to be predicted.
        :param batch_size: batch size per thread. Default: 4.
        :return: predicted result.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, data, batch_size):
        """
        Evaluate model.

        :param data: evaluation data.
        :param batch_size: batch size per thread.
        :return: evaluation result as a dictionary of {'metric name': metric value}
        """
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        """
        Get the trained model

        :return: Trained model
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, model_path):
        """
        Save model to model_path

        :param model_path: path to save the trained model.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, model_path):
        """
        Load existing model from model_path

        :param model_path: Path to the existing model.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self):
        """
        Releases resources.

        :return:
        """
        pass
