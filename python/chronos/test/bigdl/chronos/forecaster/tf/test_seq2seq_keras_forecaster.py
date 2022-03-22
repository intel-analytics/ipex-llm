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

from bigdl.chronos.forecaster.tf.seq2seq_forecaster import Seq2SeqForecaster
from unittest import TestCase
import numpy as np


def create_data():
    train_num_samples = 1000
    test_num_samples = 400
    input_feature_num = 10
    output_feature_num = 2
    past_seq_len = 10
    future_seq_len = 2
    
    def get_x_y(num_sample):
        x = np.random.randn(num_sample, past_seq_len, input_feature_num)
        y = np.random.randn(num_sample, future_seq_len, output_feature_num)
        return x, y
    
    train_data = get_x_y(train_num_samples)
    test_data = get_x_y(test_num_samples)
    return train_data, test_data


class TestSeq2SeqModel(TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_seq2seq_fit_predict_evaluate(self):
        train_data, test_data = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=10,
                                       future_seq_len=2,
                                       input_feature_num=10,
                                       output_feature_num=2)
        forecaster.fit(train_data,
                       epochs=2,
                       batch_size=32)
        yhat = forecaster.predict(test_data[0])
        assert yhat.shape == (400, 2, 2)
        mse = forecaster.evaluate(test_data, multioutput="raw_values")
        assert mse[0].shape == test_data[-1].shape[1:]

    def test_seq2seq_save_load(self):
        train_data, test_data = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=10,
                                       future_seq_len=2,
                                       input_feature_num=10,
                                       output_feature_num=2)
        forecaster.fit(train_data,
                       epochs=2,
                       batch_size=32)
        yhat = forecaster.predict(test_data[0])
        with tempfile.TemporaryDirectory() as checkpoint_file:
            forecaster.save(checkpoint_file)
            forecaster.load(checkpoint_file)
            from bigdl.chronos.model.tf2.Seq2Seq_keras import LSTMSeq2Seq
            assert isinstance(forecaster.internal, LSTMSeq2Seq)
        load_model_yhat = forecaster.predict(test_data[0])
        assert yhat.shape == (400, 2, 2)
        np.testing.assert_almost_equal(yhat, load_model_yhat, decimal=5)


if __name__ == '__main__':
    pytest.main([__file__])
