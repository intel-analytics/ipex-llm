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


class Pipeline(ABC):
    """
    The pipeline object which is used to store the series of transformation of features and model
    """

    @abstractmethod
    def evaluate(self, input_df, metric=None):
        """
        evaluate the pipeline
        :param input_df: input data frame
        :param metric: the evaluation metric
        :return:
        """
        pass

    @abstractmethod
    def predict(self, input_df):
        """
        predict using the pipeline
        :param input_df: input data frame
        :return: the prediction result
        """
        pass

    @abstractmethod
    def save(self, file):
        """
        save the pipeline to a file
        :param file: the pipeline file
        :return: a pipeline object
        """
        pass
