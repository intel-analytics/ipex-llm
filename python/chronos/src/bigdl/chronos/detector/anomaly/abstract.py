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


class AnomalyDetector(ABC):
    """
    The Anomaly Detector Interface
    """
    @abstractmethod
    def fit(self, y):
        """
        fit the model to the input time series
        :param y: input time series
        """
        pass

    @abstractmethod
    def score(self):
        """
        calculate anomaly scores for each sample in input time series
        :return: anomaly scores
        """
        pass

    @abstractmethod
    def anomaly_indexes(self):
        """
        gets the indexes of the anomalies.
        :return: the indexes of the anomalies.
        """
        pass
