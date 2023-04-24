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
from bigdl.nano.utils.common import invalidInputError

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


def estimate_pattern_th(y,
                        yhat,
                        mode="default",
                        ratio=0.01,
                        dist_measure=EuclideanDistance()):
    """
    Estimate the absolute distance threshold based on y and y_hat.

    :param y: actual values
    :param yhat: predicted values
    :param mode: types of ways to find threshold
        "default" : fit data to a uniform distribution (the percentile way)
        "gaussian": fit data to a gaussian distribution
    :param ratio: the ratio of anomaly to consider as anomaly.
    :param dist_measure: measure of distance
    :return: the threshold
    """
    invalidInputError(y.shape == yhat.shape, "y shape doesn't match yhat shape")
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
        invalidInputError(False, f"Does not support ${mode}")


def estimate_trend_th(y,
                      mode="default",
                      ratio=0.01):
    """
    Estimate the min and max threshold based on y.

    :param y: actual values
    :param mode: types of ways to find threshold
        "default" : fit data to a uniform distribution (the percentile way)
        "gaussian": fit data to a gaussian distribution
    :param ratio: the ratio of anomaly to consider as anomaly.
    :return: tuple, the threshold (min, max)
    """
    if mode == "default":
        max_threshold = np.percentile(y, (1 - ratio) * 100)
        min_threshold = np.percentile(y, ratio * 100)
        return (min_threshold, max_threshold)
    elif mode == "gaussian":
        from scipy.stats import norm
        mu, sigma = norm.fit(y)
        max_t = norm.ppf(1 - ratio)
        min_t = norm.ppf(ratio)
        return (min_t * sigma + mu, max_t * sigma + mu)
    else:
        invalidInputError(False, f"Does not support ${mode}")


def detect_pattern_anomaly(y, yhat, th, dist_measure):
    anomaly_indexes = []
    for i, (y_i, yhat_i) in enumerate(zip(y, yhat)):
        if dist_measure.abs_dist(y_i, yhat_i) > th:
            anomaly_indexes.append(i)
    return anomaly_indexes


def detect_trend_anomaly(y, th):
    threshold_min = np.full_like(y, fill_value=th[0])
    threshold_max = np.full_like(y, fill_value=th[1])
    return detect_trend_anomaly_arr(y, (threshold_min, threshold_max))


def detect_trend_anomaly_arr(y, th_arr):
    min_diff = y - th_arr[0]
    max_diff = y - th_arr[1]
    anomaly_indexes = np.logical_or(min_diff < 0, max_diff > 0)
    anomaly_scores = np.zeros_like(y)
    anomaly_scores[anomaly_indexes] = 1
    return list(set(np.where(anomaly_scores > 0)[0]))


def detect_anomaly(y,
                   yhat=None,
                   pattern_th=math.inf,
                   trend_th=(-math.inf, math.inf),
                   dist_measure=EuclideanDistance()):
    """
    Detect anomalies. Each sample can have 1 or more dimensions.

    :param y: the values to detect. shape could be 1-D (num_samples,)
        or 2-D array (num_samples, features)
    :param yhat: the predicted values, a tensor with same shape as y,
        default to be None
    :param pattern_th: a single value, specify absolute distance threshold
                       between y and yhat
    :param trend_th: a tuple composed of min_threshold and max_threshold,
                     specify min and max threshold for y
    :param dist_measure: measure of distance
    :return: dict, the anomaly values indexes in the samples, including pattern and trend type
    """
    pattern_anomaly_indexes, trend_anomaly_indexes = [], []
    pattern_anomaly_scores, trend_anomaly_scores = np.zeros_like(y), np.zeros_like(y)
    # Detect pattern anomaly
    if yhat is not None:
        invalidInputError(isinstance(pattern_th, int) or isinstance(pattern_th, float),
                          f"Pattern threshold format {type(pattern_th)} is not supported, "
                          "please specify int or float value.")
        pattern_anomaly_indexes = detect_pattern_anomaly(y, yhat, pattern_th, dist_measure)

    # Detect trend anomaly
    invalidInputError(isinstance(trend_th, tuple) and len(trend_th) == 2,
                      "Trend threshold is supposed to be a tuple of two elements.")
    if (isinstance(trend_th[0], int) or isinstance(trend_th[0], float)) \
            and (isinstance(trend_th[1], int) or isinstance(trend_th[1], float)):
        # min / max values are scalars
        invalidInputError(trend_th[0] <= trend_th[1],
                          "Trend threshold is composed of (min, max), max should not be smaller.")
        trend_anomaly_indexes = detect_trend_anomaly(y, trend_th)
    elif trend_th[0].shape == y.shape and trend_th[1].shape == y.shape:
        # min max values are arrays
        invalidInputError(np.all((trend_th[1] - trend_th[0]) >= 0),
                          "In trend threshold (min, max), each data point in max tensor"
                          " should not be smaller.")
        trend_anomaly_indexes = detect_trend_anomaly_arr(y, trend_th)
    else:
        invalidInputError(False, f"Threshold format ${str(trend_th)} is not supported")

    pattern_anomaly_scores[pattern_anomaly_indexes] = 1
    trend_anomaly_scores[trend_anomaly_indexes] = 1

    anomaly_indexes = list(set(pattern_anomaly_indexes + trend_anomaly_indexes))
    anomaly_scores = np.zeros_like(y)
    anomaly_scores[anomaly_indexes] = 1

    index_dict = {'pattern anomaly index': pattern_anomaly_indexes,
                  'trend anomaly index': trend_anomaly_indexes,
                  'anomaly index': anomaly_indexes}
    score_dict = {'pattern anomaly score': pattern_anomaly_scores,
                  'trend anomaly score': trend_anomaly_scores,
                  'anomaly score': anomaly_scores}

    return index_dict, score_dict


class ThresholdDetector(AnomalyDetector):
    """
        Example:
            >>> #The dataset is split into x_train, x_test, y_train, y_test
            >>> forecaster = Forecaster(...)
            >>> forecaster.fit(x=x_train, y=y_train, ...)
            >>> y_pred = forecaster.predict(x_test)
            >>> td = ThresholdDetector()
            >>> td.fit(y_test, y_pred)
            >>> anomaly_scores = td.score()
            >>> anomaly_indexes = td.anomaly_indexes()
    """

    def __init__(self):
        """
        Initialize a ThresholdDetector.
        """
        # to detect pattern anomaly, specify absolute distance threshold between y_true and y_pred
        self.pattern_th = math.inf
        # to detect trend anomaly, specify min and max threshold for y_true
        self.trend_th = (-math.inf, math.inf)
        self.ratio = 0.01
        self.dist_measure = EuclideanDistance()
        self.mode = "default"
        self.anomaly_indexes_ = None
        self.anomaly_scores_ = None

    def set_params(self,
                   mode="default",
                   ratio=0.01,
                   pattern_threshold=math.inf,
                   trend_threshold=(-math.inf, math.inf),
                   dist_measure=EuclideanDistance()):
        """
        Set parameters for ThresholdDetector

        :param mode: mode can be "default" or "gaussian".
            "default" : fit data according to a uniform distribution
            "gaussian": fit data according to a gaussian distribution
        :param ratio: the ratio of anomaly to consider as anomaly.
        :param pattern_threshold: a single value, specify absolute distance threshold between real
            data and predicted data to detect pattern anomaly.
        :param trend_threshold: a tuple composed of min_threshold and max_threshold, specify min
            and max threshold for real data to detect trend anomaly.
        :param dist_measure: measure of distance
        """
        self.ratio = ratio
        self.dist_measure = dist_measure
        self.mode = mode
        self.pattern_th = pattern_threshold
        invalidInputError(isinstance(trend_threshold, tuple) and len(trend_threshold) == 2,
                          "Trend threshold is supposed to be a tuple of two elements.")
        self.trend_th = trend_threshold

    def fit(self, y, y_pred=None):
        """
        Fit the model

        :param y: the values to detect. shape could be 1-D (num_samples,)
            or 2-D array (num_samples, features)
        :param y_pred: the predicted values, a tensor with same shape as y,
            default to be None.
        """
        if not isinstance(self.trend_th[0], np.ndarray) and self.trend_th[0] == -math.inf and \
                not isinstance(self.trend_th[1], np.ndarray) and self.trend_th[1] == math.inf:
            self.trend_th = estimate_trend_th(y,
                                              mode=self.mode,
                                              ratio=self.ratio)
        if y_pred is not None and self.pattern_th == math.inf:
            self.pattern_th = estimate_pattern_th(y,
                                                  y_pred,
                                                  mode=self.mode,
                                                  ratio=self.ratio,
                                                  dist_measure=self.dist_measure)
        # calculate anomalies in advance in case score does not specify input
        anomalies = detect_anomaly(y, y_pred, self.pattern_th, self.trend_th, self.dist_measure)
        self.anomaly_indexes_ = anomalies[0]
        self.anomaly_scores_ = anomalies[1]

    def score(self, y=None, y_pred=None):
        """
        Gets the anomaly scores for each sample. Each anomaly score is either 0 or 1,
        where 1 indicates an anomaly.

        :param y: new time series to detect anomaly. If y is None, returns anomalies in y_pred.
            Moreover, if both y and y_hat are None, returns anomalies in the fit input.
        :param y_pred: predicted values corresponding to y

        :return: dict, anomaly score for each sample composed of pattern and trend type,
            in an array format with the same size as input
        """
        if self.anomaly_scores_ is None:
            invalidInputError(False, "please call fit before calling score")

        if y is None and y_pred is None:
            return self.anomaly_scores_
        elif y is None:
            _, score_dict = detect_anomaly(y=y_pred,
                                           trend_th=self.trend_th,
                                           dist_measure=self.dist_measure)
            return score_dict
        else:
            _, score_dict = detect_anomaly(y, yhat=y_pred,
                                           pattern_th=self.pattern_th,
                                           trend_th=self.trend_th,
                                           dist_measure=self.dist_measure)
            return score_dict

    def anomaly_indexes(self, y=None, y_pred=None):
        """
        Gets the indexes of the anomalies.

        :param y: new time series to detect anomaly. If y is None, returns anomalies in y_pred.
            Moreover, if both y and y_hat are None, returns anomalies in the fit input.
        :param y_pred: predicted values corresponding to y

        :return: dict, anomaly indexes composed of pattern and trend type
        """
        if self.anomaly_indexes_ is None:
            invalidInputError(False, "please call fit before calling score")

        if y is None and y_pred is None:
            return self.anomaly_indexes_
        elif y is None:
            index_dict, _ = detect_anomaly(y=y_pred,
                                           trend_th=self.trend_th,
                                           dist_measure=self.dist_measure)
            return index_dict
        else:
            index_dict, _ = detect_anomaly(y, yhat=y_pred,
                                           pattern_th=self.pattern_th,
                                           trend_th=self.trend_th,
                                           dist_measure=self.dist_measure)
            return index_dict
