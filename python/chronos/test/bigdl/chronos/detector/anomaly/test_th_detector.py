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

from bigdl.chronos.forecaster.lstm_forecaster import LSTMForecaster
from bigdl.chronos.detector.anomaly import ThresholdDetector


class TestThresholdDetector(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

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
        return np.array(X).astype(np.float32), np.array(Y).astype(np.float32)

    def test_fit_score(self):
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
        y_train = np.expand_dims(y_train, 1)
        forecaster = LSTMForecaster(past_seq_len=look_back,
                                    input_feature_num=x_train.shape[-1],
                                    output_feature_num=1,
                                    hidden_dim=32,
                                    layer_num=2)
        forecaster.fit(data=(x_train, y_train), batch_size=1024, epochs=50)
        y_predict = forecaster.predict(x_test)
        y_predict = np.squeeze(y_predict, axis=1)

        # find anomaly using a manual set threshold
        td = ThresholdDetector()
        td.set_params(threshold=10)
        td.fit(y_test, y_predict)
        anomaly_scores = td.score()
        assert len(list(np.where(anomaly_scores > 0)[0])) == 0
        anomaly_indexes = td.anomaly_indexes()
        assert len(anomaly_indexes) == 0

        # if threshold is not provided, ThresholdDetector can fit to the data
        ratio = 0.1
        td = ThresholdDetector()
        td.set_params(ratio=ratio)
        td.fit(y_test, y_predict)
        fitted_anomaly_indexes = td.anomaly_indexes()
        assert len(fitted_anomaly_indexes) == int(ratio * y_test.shape[0])

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
        td.set_params(threshold=3)
        td.fit(y_test, y_pred)
        anomaly_scores = td.score()
        assert len(set(np.where(anomaly_scores > 0)[0])) == num_anomaly
        anomaly_indexes = td.anomaly_indexes()
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
        td.set_params(threshold=(-1, 1))
        td.fit(y_test)
        anomaly_scores = td.score()
        assert len(set(np.where(anomaly_scores > 0)[0])) == num_anomaly
        anomaly_indexes = td.anomaly_indexes()
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
        assert abs(td.th - (norm.ppf(1 - ratio) * sigma + mu)) < 0.04

    def test_corner_cases(self):
        td = ThresholdDetector()
        with pytest.raises(RuntimeError):
            td.score()
        with pytest.raises(RuntimeError):
            td.anomaly_indexes()

        time = np.arange(0, 1, 0.5)
        y = np.sin(time)
        td.set_params(mode="dummy")
        with pytest.raises(ValueError):
            td.fit(y, y)
        td.set_params(mode="gaussian")
        with pytest.raises(ValueError):
            td.fit(y)
        td.set_params(threshold="1")
        with pytest.raises(ValueError):
            td.fit(y)
        td.set_params(threshold=(1, -1))
        with pytest.raises(ValueError):
            td.fit(y)
        td.set_params(threshold=(np.array([-1]), np.array([-1])))
        with pytest.raises(ValueError):
            td.fit(y)
        td.set_params(threshold=(np.array([1, 1]), np.array([-1, -1])))
        with pytest.raises(ValueError):
            td.fit(y)

if __name__ == "__main__":
    pytest.main([__file__])
