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

from bigdl.chronos.detector.anomaly.abstract import AnomalyDetector
from bigdl.chronos.detector.anomaly.util import INTEL_EXT_DBSCAN

import numpy as np


class DBScanDetector(AnomalyDetector):
    """
        Example:
            >>> #The dataset to detect is y
            >>> y = numpy.array(...)
            >>> ad = DBScanDetector(eps=0.1, min_samples=6)
            >>> ad.fit(y)
            >>> anomaly_scores = ad.score()
            >>> anomaly_indexes = ad.anomaly_indexes()
    """

    def __init__(self,
                 eps=0.01,
                 min_samples=6,
                 **argv):
        """
        Initialize a DBScanDetector.

        :param eps: The maximum distance between two samples for one to be considered
            as the neighborhood of the other.
            It is a parameter of DBSCAN, refer to sklearn.cluster.DBSCAN docs for more details.
        :param min_samples: The number of samples (or total weight) in a neighborhood
            for a point to be considered as a core point.
            It is a parameter of DBSCAN, refer to sklearn.cluster.DBSCAN docs for more details.
        :param argv: Other parameters used in DBSCAN.
            Refer to sklearn.cluster.DBSCAN docs for more details.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.argv = argv
        self.anomaly_indexes_ = None
        self.anomaly_scores_ = None

    def check_data(self, arr):
        if len(arr.shape) > 1:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False,
                              "Only univariate time series is supported")

    def fit(self, y, use_sklearnex=True):
        """
        Fit the model

        :param y: the input time series. y must be 1-D numpy array.
        :param use_sklearnex: bool, If scikit-learn-intelex is not installed,
               DBScanDetector will fallback to use stock sklearn.
        """
        self.check_data(y)
        self.anomaly_scores_ = np.zeros_like(y)

        with INTEL_EXT_DBSCAN(use_sklearnex=use_sklearnex,
                              algorithm_list=['DBSCAN']) as DBSCAN:
            clusters = DBSCAN(eps=self.eps, min_samples=self.min_samples)\
                .fit(y.reshape(-1, 1), **self.argv)
        labels = clusters.labels_
        outlier_indexes = np.where(labels == -1)[0]
        self.anomaly_indexes_ = outlier_indexes
        self.anomaly_scores_[self.anomaly_indexes_] = 1

    def score(self):
        """
        Gets the anomaly scores for each sample.
        Each anomaly score is either 0 or 1, where 1 indicates an anomaly.

        :return: anomaly score for each sample, in an array format with the same size as input
        """
        if self.anomaly_indexes_ is None:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False,
                              "Please call fit first")
        return self.anomaly_scores_

    def anomaly_indexes(self):
        """
        Gets the indexes of the anomalies.

        :return: the indexes of the anomalies.
        """
        if self.anomaly_indexes_ is None:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False,
                              "Please call fit first")
        return self.anomaly_indexes_
