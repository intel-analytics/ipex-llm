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
import tempfile

from unittest import TestCase
import numpy as np
import tensorflow as tf


def create_data():
    train_num_samples = 1000
    test_num_samples = 400
    input_feature_num = 10
    output_feature_num = 2
    past_seq_len = 10
    
    def get_x_y(num_sample):
        x = np.random.randn(num_sample, past_seq_len, input_feature_num)
        y = np.random.randn(num_sample, 1, output_feature_num)
        return x, y
    
    train_data = get_x_y(train_num_samples)
    test_data = get_x_y(test_num_samples)
    return train_data, test_data


@pytest.mark.skipif(tf.__version__ < '2.0.0', reason="Run only when tf > 2.0.0.")
class TestLSTMForecaster(TestCase):
    def setUp(self):
        from bigdl.chronos.forecaster.tf.lstm_forecaster import LSTMForecaster
        self. forecaster = LSTMForecaster(past_seq_len=10,
                                          input_feature_num=10,
                                          output_feature_num=2)

    def tearDown(self):
        pass

    def test_lstm_forecaster_fit_predict_evaluate(self):
        train_data, test_data = create_data()
        self.forecaster.fit(train_data,
                       epochs=2,
                       batch_size=32)
        yhat = self.forecaster.predict(test_data[0],
                                  batch_size=32)
        assert yhat.shape == (400, 1, 2)
        mse = self.forecaster.evaluate(test_data,
                                  batch_size=32,
                                  multioutput="raw_values")
        assert mse[0].shape == test_data[1].shape[1:]

    def test_lstm_forecaster_save_load(self):
        train_data, test_data = create_data()
        self.forecaster.fit(train_data, epochs=2)
        yhat = self.forecaster.predict(test_data[0])
        with tempfile.TemporaryDirectory() as checkpoint_file:
            self.forecaster.save(checkpoint_file)
            self.forecaster.load(checkpoint_file)
            from bigdl.chronos.model.tf2.VanillaLSTM_keras import LSTMModel
            assert isinstance(self.forecaster.internal, LSTMModel)
        load_model_yhat = self.forecaster.predict(test_data[0])
        assert yhat.shape == (400, 1, 2)
        np.testing.assert_almost_equal(yhat, load_model_yhat, decimal=5)

if __name__ == '__main__':
    pytest.main([__file__])
