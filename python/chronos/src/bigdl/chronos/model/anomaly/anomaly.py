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

import math
import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod


class Distance(metaclass=ABCMeta):
    """
    The Base Distance Class.
    """
    @abstractmethod
    def distance(self, x, y):
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

    def distance(self, x, y):
        return np.linalg.norm(x - y)


class ThresholdEstimator:
    """
    An estimator to find the proper threshold.
    """
    def fit(self,
            y,
            yhat,
            mode="default",
            ratio=0.01,
            dist_measure=EuclideanDistance()
            ):
        """
        fit the y and yhat and find the proper threshold
        :param y: actual values
        :param yhat: predicted values
        :param mode: types of ways to find threshold
            "default" : fit data to a uniform distribution (the percentile way)
            "gaussian": fit data to a gaussian distribution *TBD
        :param ratio: the ratio of anomaly to consider as anomaly.
        :return: the threshold
        """
        assert y.shape == yhat.shape
        diff = [dist_measure.distance(m, n) for m, n in zip(y, yhat)]
        if mode == "default":
            threshold = np.percentile(diff, (1-ratio)*100)
            return threshold
        elif mode == "gaussian":
            from scipy.stats import norm
            mu, sigma = norm.fit(diff)
            t = norm.ppf(1-ratio)
            return t*sigma+mu
        else:
            raise Exception("Does not support", mode)


class DetectorBase(metaclass=ABCMeta):
    """
    Base class for detector
    """
    @abstractmethod
    def detect(self, y, **kwargs):
        """
        Detect anomalies in dataset
        :param y: the dataset
        :return: the anomaly indexes in y
        """
        pass


class ThresholdDetector(DetectorBase):
    """
    Anomaly detector
    """

    def __init__(self):
        pass

    def detect(self,
               y,
               yhat=None,
               threshold=math.inf,
               dist_measure=EuclideanDistance()):
        """
        Detect anomalies. Each sample can have 1 or more dimensions.
        :param dist_measure:
        :param y: the values to detect. shape could be
                1-D (num_samples,)
                or 2-D array (num_samples, features)
        :param yhat: the estimated values, a tensor with same shape as y,
                could be None when threshold is a tuple
        :param threshold: threshold, could be
            1. a single value -  absolute distance threshold, same for all samples
            2. a 1-D array in shape (num_samples,) - per sample absolute distance threshold
            3. a tensor in same shape as y and yhat - per dimension absolute distance threshold
            4. a tuple (min, max) min and max tensors, same shape as y, yhat is ignored in this case
        :return: the anomaly values indexes in the samples, i.e. num_samples dimension.
        """
        self.threshold = threshold
        self.dist_measure = dist_measure
        if isinstance(threshold, int) or \
                isinstance(threshold, float):
            return self._check_all_distance(y, yhat)
        elif isinstance(threshold, np.ndarray):
            if len(threshold.shape) == 2:
                self._check_per_dim_distance(y, yhat, threshold)
            elif len(threshold.shape) == 1:
                self._check_per_sample_distance(y, yhat, threshold)
            else:
                raise ValueError(
                    "threshold shape", str(
                        threshold.shape), "is not valid")
        elif isinstance(threshold, tuple) \
                and len(threshold) == 2 \
                and threshold[0].shape == y.shape \
                and threshold[-1].shape == y.shape:
            return self._check_range(y, threshold)
        else:
            raise ValueError(
                "threshold shape", str(threshold),
                "is not valid")

    def _check_all_distance(self, y, yhat):
        index = []
        for i in range(y.shape[0]):
            diff = self.dist_measure.distance(y[i], yhat[i])
            if diff >= self.threshold:
                index.append(i)
        return index

    def _check_per_dim_distance(self, y, yhat, threshold):
        raise NotImplementedError("Does not support check per dim distance")

    def _check_per_sample_distance(self, y, yhat, threshold):
        raise NotImplementedError("Does not support check per sample distance")

    def _check_range(self, y, threshold):
        min_diff = y - threshold[0]
        anomaly_index = set(np.where(min_diff < 0)[0])
        max_diff = y - threshold[1]
        anomaly_index.update(np.where(max_diff > 0)[0])
        return list(anomaly_index)
