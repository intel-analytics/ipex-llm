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

import pytest
import numpy as np
import pandas as pd
from unittest import TestCase

from bigdl.chronos.utils import LazyImport
TCNForecaster = LazyImport('bigdl.chronos.forecaster.TCNForecaster')
from bigdl.chronos.detector.anomaly import ThresholdDetector
from ... import op_torch


@op_torch
class TestThresholdDetector(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def create_data(self):
        num_train_samples = 1000
        num_test_samples = 400
        input_time_steps = 24
        input_feature_dim = 1
        output_time_steps = 5
        output_feature_dim = 1

        def get_x_y(num_samples):
            x = np.random.rand(num_samples, input_time_steps, input_feature_dim).astype(np.float32)
            y = x[:, -output_time_steps:, :]*2 + \
                np.random.rand(num_samples, output_time_steps, output_feature_dim).astype(np.float32)
            return x, y

        train_data = get_x_y(num_train_samples)
        test_data = get_x_y(num_test_samples)

        return train_data, test_data

    def test_fit_score(self):
        train_data, test_data = self.create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                    future_seq_len=5,
                                    input_feature_num=1,
                                    output_feature_num=1)
        forecaster.fit(train_data, epochs=1)
        y_test_pred = forecaster.predict(test_data[0])
        y_test = test_data[1]

        # find anomaly using a manual set threshold
        td = ThresholdDetector()
        td.set_params(pattern_threshold=10, trend_threshold=(-3,3))
        td.fit(y_test, y_test_pred)
        anomaly_scores = td.score()['anomaly score']
        assert len(list(np.where(anomaly_scores > 0)[0])) == 0
        anomaly_indexes = td.anomaly_indexes()['anomaly index']
        assert len(anomaly_indexes) == 0

    def test_threshold_single(self):
        sample_num = 10
        feature_dim = 5
        num_anomaly = 5
        # predicted value
        y_pred = np.full((sample_num, feature_dim), 0)
        # actual value
        y_test = np.full(sample_num * feature_dim, 0.2)

        gen_rand_indexes = [0, 7, 16, 33, 45]
        y_test[gen_rand_indexes] = 10
        y_test = y_test.reshape((sample_num, feature_dim))

        td = ThresholdDetector()
        td.set_params(pattern_threshold=3)
        td.fit(y_test, y_pred)
        anomaly_scores = td.score()['anomaly score']
        assert len(set(np.where(anomaly_scores > 0)[0])) == num_anomaly
        anomaly_indexes = td.anomaly_indexes()['anomaly index']
        assert len(anomaly_indexes) == num_anomaly

    def test_threshold_minmax(self):
        sample_num = 10
        feature_dim = 5
        num_anomaly = 5
        # actual value
        y_test = np.zeros(sample_num * feature_dim)

        gen_rand_indexes = [0, 7, 16, 33, 45]
        y_test[gen_rand_indexes] = 10
        y_test = y_test.reshape((sample_num, feature_dim))

        td = ThresholdDetector()
        td.set_params(trend_threshold=(-1, 1))
        td.fit(y_test)
        anomaly_scores = td.score()['anomaly score']
        assert len(set(np.where(anomaly_scores > 0)[0])) == num_anomaly
        anomaly_indexes = td.anomaly_indexes()['anomaly index']
        assert len(anomaly_indexes) == num_anomaly

    def test_mode_gaussian(self):
        sample_num = 500
        # actual value
        y_test = np.full(sample_num, 2)
        mu, sigma, ratio = 3, 0.1, 0.01
        s = np.random.normal(mu, sigma, sample_num)
        y_pred = y_test + s

        td = ThresholdDetector()
        td.set_params(mode="gaussian", ratio=ratio)
        td.fit(y_test, y_pred)
        # check estimated threshold
        from scipy.stats import norm
        assert abs(td.pattern_th - (norm.ppf(1 - ratio) * sigma + mu)) < 0.04

    def test_corner_cases(self):
        td = ThresholdDetector()
        with pytest.raises(RuntimeError):
            td.score()
        with pytest.raises(RuntimeError):
            td.anomaly_indexes()

        time = np.arange(0, 1, 0.5)
        y = np.sin(time)
        with pytest.raises(RuntimeError):
            td.set_params(mode="dummy")
            td.fit(y, y)
        with pytest.raises(RuntimeError):
            td.set_params(pattern_threshold="1")
            td.fit(y, y)
        with pytest.raises(RuntimeError):
            td.set_params(pattern_threshold=(1, -1))
            td.fit(y, y)
        with pytest.raises(RuntimeError):
            td.set_params(pattern_threshold=(np.array([-1]), np.array([-1])))
            td.fit(y, y)
        with pytest.raises(RuntimeError):
            td.set_params(pattern_threshold=(np.array([1, 1]), np.array([-1, -1])))
            td.fit(y, y)
        with pytest.raises(RuntimeError):
            td.set_params(trend_threshold="1")
            td.fit(y)
        with pytest.raises(RuntimeError):
            td.set_params(trend_threshold=(1, -1))
            td.fit(y)
        with pytest.raises(RuntimeError):
            td.set_params(trend_threshold=(np.array([-1]), np.array([-1])))
            td.fit(y)
        with pytest.raises(RuntimeError):
            td.set_params(trend_threshold=(np.array([1, 1]), np.array([-1, -1])))
            td.fit(y)

if __name__ == "__main__":
    pytest.main([__file__])
