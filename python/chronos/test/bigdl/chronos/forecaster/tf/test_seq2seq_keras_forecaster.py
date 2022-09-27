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
import os

from unittest import TestCase
import numpy as np
import tensorflow as tf
from bigdl.chronos.forecaster.tf.seq2seq_forecaster import Seq2SeqForecaster


def create_data(tf_data=False, batch_size=32):
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

    if tf_data:
        from_tensor_slices = tf.data.Dataset.from_tensor_slices
        train_data = from_tensor_slices(train_data).cache()\
                                                   .shuffle(train_num_samples)\
                                                   .batch(batch_size)\
                                                   .prefetch(tf.data.AUTOTUNE)
        test_data = from_tensor_slices(test_data).cache()\
                                                 .batch(batch_size)\
                                                 .prefetch(tf.data.AUTOTUNE)
    return train_data, test_data

def create_tsdataset(roll=True):
    from bigdl.chronos.data import TSDataset
    import pandas as pd
    timeseries = pd.date_range(start='2020-01-01', freq='D', periods=1000)
    df = pd.DataFrame(np.random.rand(1000, 2),
                      columns=['value1', 'value2'],
                      index=timeseries,
                      dtype=np.float32)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'timeseries'}, inplace=True)
    train, _, test = TSDataset.from_pandas(df=df,
                                           dt_col='timeseries',
                                           target_col=['value1', 'value2'],
                                           with_split=True)
    if roll:
        for tsdata in [train, test]:
            tsdata.roll(lookback=24, horizon=2)
    return train, test


@pytest.mark.skipif(tf.__version__ < '2.0.0', reason="Run only when tf > 2.0.0.")
class TestSeq2SeqForecaster(TestCase):
    
    def setUp(self):
        from bigdl.chronos.forecaster.tf.seq2seq_forecaster import Seq2SeqForecaster
        self.forecaster = Seq2SeqForecaster(past_seq_len=10,
                                            future_seq_len=2,
                                            input_feature_num=10,
                                            output_feature_num=2)

    def tearDown(self):
        del self.forecaster

    def test_seq2seq_fit_predict_evaluate(self):
        train_data, test_data = create_data()
        self.forecaster.fit(train_data,
                            epochs=2,
                            batch_size=32)
        yhat = self.forecaster.predict(test_data[0])
        assert yhat.shape == (400, 2, 2)
        mse = self.forecaster.evaluate(test_data, multioutput="raw_values")
        assert mse[0].shape == test_data[-1].shape[1:]
    
    def test_seq2seq_fit_tf_data(self):
        train_data, test_data = create_data(tf_data=True)
        self.forecaster.fit(train_data,
                            epochs=2)
        yhat = self.forecaster.predict(test_data)
        assert yhat.shape == (400, 2, 2)

    def test_seq2seq_save_load(self):
        train_data, test_data = create_data()
        self.forecaster.fit(train_data,
                            epochs=2,
                            batch_size=32)
        yhat = self.forecaster.predict(test_data[0])
        with tempfile.TemporaryDirectory() as tmp_dir_file:
            tmp_dir_file = os.path.join(tmp_dir_file, 'seq2seq.ckpt')
            self.forecaster.save(tmp_dir_file)
            self.forecaster.load(tmp_dir_file)
            from bigdl.chronos.model.tf2.Seq2Seq_keras import LSTMSeq2Seq
            assert isinstance(self.forecaster.internal, LSTMSeq2Seq)
        load_model_yhat = self.forecaster.predict(test_data[0])
        assert yhat.shape == (400, 2, 2)
        np.testing.assert_almost_equal(yhat, load_model_yhat, decimal=5)

    def test_s2s_customized_loss_metric(self):
        train_data, test_data = create_data(tf_data=True)
        loss = tf.keras.losses.MeanSquaredError()
        def customized_metric(y_true, y_pred):
            return tf.keras.losses.MeanSquaredError(tf.convert_to_tensor(y_pred),
                                      tf.convert_to_tensor(y_true)).numpy()
        from bigdl.chronos.forecaster.tf.seq2seq_forecaster import Seq2SeqForecaster
        self.forecaster = Seq2SeqForecaster(past_seq_len=10,
                                            future_seq_len=2,
                                            input_feature_num=10,
                                            output_feature_num=2,
                                            loss=loss,
                                            metrics=[customized_metric],
                                            lr=0.01)
        self.forecaster.fit(train_data, epochs=2, batch_size=32)
        yhat = self.forecaster.predict(test_data)
        with tempfile.TemporaryDirectory() as tmp_dir_file:
            tmp_dir_file = os.path.join(tmp_dir_file, 'seq2seq.ckpt')
            self.forecaster.save(tmp_dir_file)
            self.forecaster.load(tmp_dir_file)
            from bigdl.chronos.model.tf2.Seq2Seq_keras import LSTMSeq2Seq
            assert isinstance(self.forecaster.internal, LSTMSeq2Seq)
        load_model_yhat = self.forecaster.predict(test_data)
        assert yhat.shape == (400, 2, 2)
        np.testing.assert_almost_equal(yhat, load_model_yhat, decimal=5)

    def test_s2s_from_tsdataset(self):
        train, test = create_tsdataset(roll=True)
        s2s = Seq2SeqForecaster.from_tsdataset(train,
                                               lstm_hidden_dim=16,
                                               lstm_layer_num=2)
        s2s.fit(train,
                epochs=2,
                batch_size=32)
        yhat = s2s.predict(test, batch_size=32)
        test.roll(lookback=s2s.model_config['past_seq_len'],
                  horizon=s2s.model_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape

        del s2s

        train, test = create_tsdataset(roll=False)
        s2s = Seq2SeqForecaster.from_tsdataset(train,
                                               past_seq_len=24,
                                               future_seq_len=2,
                                               lstm_hidden_dim=16,
                                               lstm_layer_num=2)
        s2s.fit(train,
                epochs=2,
                batch_size=32)
        yhat = s2s.predict(test, batch_size=None)
        test.roll(lookback=s2s.model_config['past_seq_len'],
                  horizon=s2s.model_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape


if __name__ == '__main__':
    pytest.main([__file__])
