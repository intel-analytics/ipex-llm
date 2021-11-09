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

import math
import numpy as np
from bigdl.chronos.detector.anomaly.abstract import AnomalyDetector

from abc import ABC, abstractmethod


class Distance(ABC):
    """
    The Base Distance Class.
    """

    @abstractmethod
    def abs_dist(self, x, y):
        """
        Calculate the distance between x and y. a and b should be in same shape.

        :param x: the first tensor
        :param y: the second tensor
        :return: the absolute distance between x and y
        """
        pass


class EuclideanDistance(Distance):
    """
    Euclidean Distance Measure
    """

    def __init__(self):
        pass

    def abs_dist(self, x, y):
        return np.linalg.norm(x - y)


def estimate_th(y,
                yhat,
                mode="default",
                ratio=0.01,
                dist_measure=EuclideanDistance()):
    """
    Estimate the threshold based on y and yhat

    :param y: actual values
    :param yhat: predicted values
    :param mode: types of ways to find threshold
        "default" : fit data to a uniform distribution (the percentile way)
        "gaussian": fit data to a gaussian distribution
    :param ratio: the ratio of anomaly to consider as anomaly.
    :param dist_measure: measure of distance
    :return: the threshold
    """
    assert y.shape == yhat.shape
    diff = [dist_measure.abs_dist(m, n) for m, n in zip(y, yhat)]
    if mode == "default":
        threshold = np.percentile(diff, (1 - ratio) * 100)
        return threshold
    elif mode == "gaussian":
        from scipy.stats import norm
        mu, sigma = norm.fit(diff)
        t = norm.ppf(1 - ratio)
        return t * sigma + mu
    else:
        raise ValueError("Does not support", mode)


def detect_all(y, yhat, th, dist_measure):
    anomaly_scores = np.zeros_like(y)
    anomaly_indexes = []
    for i, (y_i, yhat_i) in enumerate(zip(y, yhat)):
        if dist_measure.abs_dist(y_i, yhat_i) > th:
            anomaly_indexes.append(i)
            anomaly_scores[i] = 1
    return anomaly_indexes, anomaly_scores


def detect_range(y, th):
    # use threshold (-1, 1) for each dimension
    threshold_min = np.full_like(y, fill_value=th[0])
    threshold_max = np.full_like(y, fill_value=th[1])
    return detect_range_arr(y, (threshold_min, threshold_max))


def detect_range_arr(y, th_arr):
    min_diff = y - th_arr[0]
    max_diff = y - th_arr[1]
    anomaly_indexes = np.logical_or(min_diff < 0, max_diff > 0)
    anomaly_scores = np.zeros_like(y)
    anomaly_scores[anomaly_indexes] = 1
    # anomaly_index.update(np.where(max_diff > 0)[0])
    return list(set(np.where(anomaly_scores > 0)[0])), anomaly_scores


def detect_anomaly(y,
                   yhat=None,
                   th=math.inf,
                   dist_measure=EuclideanDistance()):
    """
    Detect anomalies. Each sample can have 1 or more dimensions.

    :param y: the values to detect. shape could be 1-D (num_samples,)
        or 2-D array (num_samples, features)
    :param yhat: the estimated values, a tensor with same shape as y,
        could be None when threshold is a tuple
    :param th: threshold, could be

        1. a single value - absolute distance threshold, same for all samples

        2. a tuple (min, max) - min and max are either int/float or tensors in same shape as y,
        yhat is ignored in this case
    :param dist_measure: measure of distance
    :return: the anomaly values indexes in the samples, i.e. num_samples dimension.
    """
    if isinstance(th, int) or isinstance(th, float):
        if yhat is None:
            raise ValueError("Please specify a threshold range (min,max) "
                             "if forecast values are not available")
        return detect_all(y, yhat, th, dist_measure)
    elif isinstance(th, tuple) and len(th) == 2:
        # min max values are scalars
        if (isinstance(th[0], int) or isinstance(th[0], float)) \
                and (isinstance(th[1], int) or isinstance(th[1], float)):
            if th[0] > th[1]:
                raise ValueError(
                    "In threshold (min,max), max should be larger than min")
            return detect_range(y, th)
        # min max values are arrays
        elif th[0].shape == y.shape and th[-1].shape == y.shape:
            if np.any((th[1] - th[0]) < 0):
                raise ValueError("In threshold (min,max) ",
                                 "each data point in max tensor should be larger than min")
            return detect_range_arr(y, th)
        else:
            raise ValueError("Threshold format", str(th), "is not supported")
    else:
        raise ValueError(
            "Threshold format", str(th),
            "is not supported")


class ThresholdDetector(AnomalyDetector):
    """
        Example:
            >>> #The dataset is split into x_train, x_test, y_train, y_test
            >>> forecaster = Forecaster(...)
            >>> forecaster.fit(x=x_train, y=y_train, ...)
            >>> y_pred = forecaster.predict(x_test)
            >>> td = ThresholdDetector()
            >>> td.set_params(threshold=10)
            >>> td.fit(y_test, y_pred)
            >>> anomaly_scores = td.score()
            >>> anomaly_indexes = td.anomaly_indexes()
    """

    def __init__(self):
        """
        Initialize a ThresholdDetector.
        """
        self.th = math.inf
        self.ratio = 0.01
        self.dist_measure = EuclideanDistance()
        self.mode = "default"
        self.anomaly_indexes_ = None
        self.anomaly_scores_ = None

    def set_params(self,
                   mode="default",
                   ratio=0.01,
                   threshold=math.inf,
                   dist_measure=EuclideanDistance()):
        """
        Set parameters for ThresholdDetector

        :param mode: mode can be "default" or "gaussian".
            "default" : fit data according to a uniform distribution
            "gaussian": fit data according to a gaussian distribution
        :param ratio: the ratio of anomaly to consider as anomaly.
        :param threshold: threshold, could be

            1. a single value - absolute distance threshold, same for all samples

            2. a tuple (min, max) - min and max are either int/float or tensors in same shape as y,
            yhat is ignored in this case
        :param dist_measure: measure of distance
        """
        self.ratio = ratio
        self.dist_measure = dist_measure
        self.mode = mode
        self.th = threshold

    def fit(self, y, y_pred=None):
        """
        Fit the model

        :param y: the values to detect. shape could be 1-D (num_samples,)
            or 2-D array (num_samples, features)
        :param y_pred: the estimated values, a tensor with same shape as y
            could be None when threshold is a tuple
        """
        if y_pred is not None and self.th == math.inf:
            self.th = estimate_th(y,
                                  y_pred,
                                  mode=self.mode,
                                  ratio=self.ratio,
                                  dist_measure=self.dist_measure)
        # calculate anomalies in advance in case score does not specify input
        anomalies = detect_anomaly(y, y_pred, self.th, self.dist_measure)
        self.anomaly_indexes_ = anomalies[0]
        self.anomaly_scores_ = anomalies[1]

    def score(self, y=None, y_pred=None):
        """
        Gets the anomaly scores for each sample. Each anomaly score is either 0 or 1,
        where 1 indicates an anomaly.

        :param y: new time series to detect anomaly. if y is None, returns anomalies
            in the fit input, y_pred is ignored in this case
        :param y_pred: forecasts corresponding to y

        :return: anomaly score for each sample, in an array format with the same size as input
        """
        if y is None:
            if self.anomaly_scores_ is None:
                raise RuntimeError("please call fit before calling score")
            return self.anomaly_scores_
        else:
            return detect_anomaly(y,
                                  y_pred,
                                  self.th,
                                  self.dist_measure)[1]

    def anomaly_indexes(self):
        """
        Gets the indexes of the anomalies.

        :return: the indexes of the anomalies.
        """
        if self.anomaly_indexes_ is None:
            raise RuntimeError("Please call fit first")
        return self.anomaly_indexes_
