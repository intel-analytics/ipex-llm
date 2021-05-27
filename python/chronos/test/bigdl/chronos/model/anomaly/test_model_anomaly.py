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

import pytest
import numpy as np
import pandas as pd
from test.zoo.pipeline.utils.test_utils import ZooTestCase

from zoo.chronos.model.forecast.lstm_forecaster import LSTMForecaster
from zoo.chronos.model.anomaly import ThresholdDetector, ThresholdEstimator


class TestChronosModelAnomaly(ZooTestCase):

    def gen_data(self, feature_num=6, sample_num=100):
        return pd.DataFrame(data=np.random.randn(sample_num, feature_num))

    def train_test_split(self, df, test_num, look_back):
        test_split_index = test_num + look_back + 1

        # train_df
        train_df = df[:-test_num]
        test_df = df[-test_split_index:]
        test_df = test_df.reset_index(drop=True)
        return train_df, test_df

    def roll_data(self, dataset, look_back, target_col_indexes):
        """
        Generate input samples from rolling
        """
        X, Y = [], []
        data = dataset.to_numpy()
        for i in range(len(dataset) - look_back - 1):
            X.append(data[i: (i + look_back)])
            # Y.append(dataset.iloc[i + look_back, target_col_indexes])
            Y.append(data[i + look_back][target_col_indexes])
        return np.array(X), np.array(Y)

    def test_app(self):
        look_back = 4

        # generate dataframe
        data = self.gen_data(feature_num=6, sample_num=100)
        # split train and test dataframes
        train_df, test_df = self.train_test_split(data, test_num=20, look_back=look_back)

        # roll data to generate model input
        x_train, y_train = self.roll_data(dataset=train_df, look_back=look_back,
                                          target_col_indexes=[0])
        x_test, y_test = self.roll_data(dataset=test_df, look_back=look_back,
                                        target_col_indexes=[0])

        # create model, train on train data and predict on test
        lstm_config = {"lstm_units": [32] * 2, "lr": 0.001}
        forecaster = LSTMForecaster(target_dim=1, feature_dim=x_train.shape[-1], **lstm_config)
        forecaster.fit(x=x_train, y=y_train, batch_size=1024, epochs=50, distributed=False)
        y_predict = forecaster.predict(x_test)

        # find anomaly by comparing the difference between y_predict and y_test (actual)
        threshold = 10
        detector = ThresholdDetector()
        anomaly_indexes = detector.detect(y=y_test,
                                          yhat=y_predict,
                                          threshold=threshold)
        assert len(anomaly_indexes) == 0

        # if user don't have a threshold, he can choose to use estimator
        # to find a threshold first
        ratio = 0.1
        threshold = ThresholdEstimator().fit(y=y_test, yhat=y_predict, ratio=ratio)
        fitted_anomaly_indexes = detector.detect(y=y_test, yhat=y_predict, threshold=threshold)
        assert len(fitted_anomaly_indexes) == int(ratio * y_test.shape[0])

    def test_threshold_case1_multivariant(self):
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

        anomaly_indexes = ThresholdDetector().detect(y=y_test, yhat=y_pred, threshold=3)
        assert len(anomaly_indexes) == num_anomaly

    def test_threshold_case4(self):
        sample_num = 10
        feature_dim = 5
        num_anomaly = 5
        # actual value
        y_test = np.zeros(sample_num * feature_dim)

        gen_rand_indexes = [0, 7, 16, 33, 45]
        y_test[gen_rand_indexes] = 10
        y_test = y_test.reshape((sample_num, feature_dim))

        # use threshold (-1, 1) for each dimension
        threshold_min = np.ones_like(y_test) * (-1)
        threshold_max = np.ones_like(y_test)
        anomaly_indexes = ThresholdDetector().detect(y=y_test, yhat=None,
                                                     threshold=(threshold_min, threshold_max))
        assert len(anomaly_indexes) == num_anomaly

    def test_threshold_gaussian(self):
        sample_num = 500
        # actual value
        y_test = np.full(sample_num, 2)
        mu, sigma, ratio = 3, 0.1, 0.01
        s = np.random.normal(mu, sigma, sample_num)
        y = y_test + s

        threshold = ThresholdEstimator().fit(y, y_test, mode="gaussian", ratio=ratio)
        from scipy.stats import norm
        assert abs(threshold-(norm.ppf(1-ratio)*sigma+mu)) < 0.04


if __name__ == "__main__":
    pytest.main([__file__])
