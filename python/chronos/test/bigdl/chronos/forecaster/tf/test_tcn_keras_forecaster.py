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

from bigdl.chronos.utils import LazyImport
tf = LazyImport('tensorflow')
TCNForecaster = LazyImport('bigdl.chronos.forecaster.tf.tcn_forecaster.TCNForecaster')
from test.bigdl.chronos import op_tf2, op_distributed, op_inference


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
    val_data = get_x_y(test_num_samples)
    test_data = get_x_y(test_num_samples)

    if tf_data:
        from_tensor_slices = tf.data.Dataset.from_tensor_slices
        train_data = from_tensor_slices(train_data).cache()\
                                                   .shuffle(train_num_samples)\
                                                   .batch(batch_size)\
                                                   .prefetch(tf.data.AUTOTUNE)
        val_data = from_tensor_slices(val_data).batch(batch_size)\
                                               .cache()\
                                               .prefetch(tf.data.AUTOTUNE)
        test_data = from_tensor_slices(test_data).batch(batch_size)\
                                                 .cache()\
                                                 .prefetch(tf.data.AUTOTUNE)
    return train_data, val_data, test_data


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
    train, valid, test = TSDataset.from_pandas(df=df,
                                           dt_col='timeseries',
                                           target_col=['value1', 'value2'],
                                           val_ratio=0.1,
                                           with_split=True)
    if roll:
        for tsdata in [train, valid, test]:
            tsdata.roll(lookback=24, horizon=5)
    return train, valid, test


@op_tf2
class TestTCNForecaster(TestCase):
    def setUp(self):
        from bigdl.chronos.forecaster.tf.tcn_forecaster import TCNForecaster
        self.forecaster = TCNForecaster(past_seq_len=10,
                                        future_seq_len=2,
                                        input_feature_num=10,
                                        output_feature_num=2,
                                        num_channels=[15]*7)

    def tearDown(self):
        del self.forecaster

    def test_tcn_forecaster_fit_predict_evaluate(self):
        train_data, _, test_data = create_data()
        self.forecaster.fit(train_data,
                            epochs=2,
                            batch_size=32)
        yhat = self.forecaster.predict(test_data[0],
                                       batch_size=32)
        assert yhat.shape == (400, 2, 2)
        mse = self.forecaster.evaluate(test_data,
                                       batch_size=32,
                                       multioutput="raw_values")
        assert mse[0].shape == test_data[1].shape[1:]

    def test_tcn_forecaster_fit_predict_evaluate_normalization(self):
        train_data, _, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=10,
                                   future_seq_len=2,
                                   input_feature_num=10,
                                   output_feature_num=2,
                                   num_channels=[15]*7,
                                   normalization=True)
        forecaster.fit(train_data, epochs=2, batch_size=32)
        yhat = forecaster.predict(test_data[0], batch_size=32)
        assert yhat.shape == (400, 2, 2)
        mse = forecaster.evaluate(test_data, batch_size=32, multioutput="raw_values")
        assert mse[0].shape == test_data[1].shape[1:]

    def test_tcn_forecaster_evaluate(self):
        train_tsdata, _, test_tsdata = create_tsdataset()
        forecaster = TCNForecaster.from_tsdataset(train_tsdata, past_seq_len=24, future_seq_len=5)
        forecaster.fit(train_tsdata, epochs=1, batch_size=32)

        # tf dataset
        test = test_tsdata.to_tf_dataset(batch_size=32)
        metrics = forecaster.evaluate(test, multioutput='uniform_average')

        # TSDataset
        metrics_tsdata = forecaster.evaluate(test_tsdata, multioutput='uniform_average')
        np.testing.assert_almost_equal(metrics, metrics_tsdata, decimal=5)

        # numpy
        test_data = test_tsdata.to_numpy()
        metrics_data = forecaster.evaluate(test_data, multioutput='uniform_average')
        np.testing.assert_almost_equal(metrics_data, metrics_tsdata, decimal=5)

    def test_tcn_forecaster_fit_tf_data(self):
        train_data, _, test_data = create_data(tf_data=True)
        self.forecaster.fit(train_data,
                            epochs=2,
                            batch_size=32)
        yhat = self.forecaster.predict(test_data)
        assert yhat.shape == (400, 2, 2)

    def test_tcn_forecaster_save_load(self):
        train_data, _, test_data = create_data()
        self.forecaster.fit(train_data, epochs=2)
        yhat = self.forecaster.predict(test_data[0])
        with tempfile.TemporaryDirectory() as tmp_dir_file:
            tmp_dir_file = os.path.join(tmp_dir_file, 'tcn.ckpt')
            self.forecaster.save(tmp_dir_file)
            self.forecaster.load(tmp_dir_file)
            from bigdl.chronos.model.tf2.TCN_keras import TemporalConvNet
            assert isinstance(self.forecaster.internal, TemporalConvNet)
        load_model_yhat = self.forecaster.predict(test_data[0])
        assert yhat.shape == (400, 2, 2)
        np.testing.assert_almost_equal(yhat, load_model_yhat, decimal=5)

    def test_tcn_customized_loss_metric(self):
        train_data, _, test_data = create_data(tf_data=True)
        loss = tf.keras.losses.MeanSquaredError()
        def customized_metric(y_true, y_pred):
            return tf.keras.losses.MeanSquaredError(tf.convert_to_tensor(y_pred),
                                                    tf.convert_to_tensor(y_true)).numpy()
        from bigdl.chronos.forecaster.tf.tcn_forecaster import TCNForecaster
        self.forecaster = TCNForecaster(past_seq_len=10,
                                        future_seq_len=2,
                                        input_feature_num=10,
                                        output_feature_num=2,
                                        num_channels=[15]*7,
                                        loss=loss,
                                        metrics=[customized_metric],
                                        lr=0.01)
        self.forecaster.fit(train_data, epochs=2, batch_size=32)
        yhat = self.forecaster.predict(test_data)
        with tempfile.TemporaryDirectory() as tmp_dir_file:
            tmp_dir_file = os.path.join(tmp_dir_file, 'tcn.ckpt')
            self.forecaster.save(tmp_dir_file)
            self.forecaster.load(tmp_dir_file)
            from bigdl.chronos.model.tf2.TCN_keras import TemporalConvNet
            assert isinstance(self.forecaster.internal, TemporalConvNet)
        load_model_yhat = self.forecaster.predict(test_data)
        assert yhat.shape == (400, 2, 2)
        np.testing.assert_almost_equal(yhat, load_model_yhat, decimal=5)

    def test_tcn_from_tsdataset(self):
        train, _, test = create_tsdataset(roll=True)

        tcn = TCNForecaster.from_tsdataset(train,
                                            num_channels=[16]*2)
        tcn.fit(train,
                 epochs=2,
                 batch_size=32)
        yhat = tcn.predict(test, batch_size=32)
        test.roll(lookback=tcn.model_config['past_seq_len'],
                  horizon=tcn.model_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape

        del tcn

        train, _, test = create_tsdataset(roll=False)
        tcn = TCNForecaster.from_tsdataset(train,
                                            past_seq_len=24,
                                            future_seq_len=5,
                                            num_channels=[16]*2)
        tcn.fit(train,
                 epochs=2,
                 batch_size=32)
        yhat = tcn.predict(test, batch_size=None)
        test.roll(lookback=tcn.model_config['past_seq_len'],
                  horizon=tcn.model_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape

    @op_distributed
    def test_tcn_forecaster_distributed(self):
        from bigdl.orca import init_orca_context, stop_orca_context
        train_data, val_data, test_data = create_data()

        init_orca_context(cores=4, memory="4g")
        forecaster = TCNForecaster(past_seq_len=10,
                                   future_seq_len=2,
                                   input_feature_num=10,
                                   output_feature_num=2,
                                   kernel_size=3,
                                   lr=1e-3,
                                   distributed=True)

        forecaster.fit(train_data, epochs=2)
        distributed_pred = forecaster.predict(test_data[0])
        distributed_eval = forecaster.evaluate(val_data)

        model = forecaster.get_model()
        from bigdl.chronos.model.tf2.TCN_keras import TemporalConvNet
        assert isinstance(model, TemporalConvNet)

        with tempfile.TemporaryDirectory() as tmp_file_name:
            name = os.path.join(tmp_file_name, "tcn.ckpt")
            test_pred_save = forecaster.predict(test_data[0])
            forecaster.save(name)
            forecaster.load(name)
            test_pred_load = forecaster.predict(test_data[0])
        np.testing.assert_almost_equal(test_pred_save, test_pred_load)

        forecaster.to_local()
        local_pred = forecaster.predict(test_data[0])
        local_eval = forecaster.evaluate(val_data)

        np.testing.assert_almost_equal(distributed_pred, local_pred, decimal=5)
        stop_orca_context()

    @op_distributed
    def test_tcn_forecaster_distributed_illegal_input(self):
        from bigdl.orca import init_orca_context, stop_orca_context

        init_orca_context(cores=4, memory="4g")
        forecaster = TCNForecaster(past_seq_len=10,
                                   future_seq_len=2,
                                   input_feature_num=2,
                                   output_feature_num=2,
                                   kernel_size=3,
                                   lr=1e-3,
                                   distributed=True)

        train_data, _, test_data = create_data(tf_data=True)
        ts_train, _, ts_test = create_tsdataset(roll=False)
        _, y_test = ts_test.roll(lookback=10, horizon=2).to_numpy()

        forecaster.fit(ts_train, epochs=2)
        yhat = forecaster.predict(ts_test)
        assert yhat.shape == y_test.shape
        res = forecaster.evaluate(ts_test)

        # illegal input
        with pytest.raises(RuntimeError):
            forecaster.fit(train_data)
        with pytest.raises(RuntimeError):
            forecaster.evaluate(test_data)

        stop_orca_context()

    @op_inference
    def test_tcn_keras_forecaster_quantization(self):
        # Capturing behaviors during `pytest -v` influence generating the valid sampling log
        # during quantization. Work around by using `pytest -s` to test this ut.
        train_data, _, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=10,
                                   future_seq_len=2,
                                   input_feature_num=10,
                                   output_feature_num=2)

        forecaster.fit(train_data, epochs=1)
        forecaster.quantize(input_data=train_data[0], target_data=train_data[1])
        assert forecaster.accelerated_model
        assert forecaster.accelerate_method == "tensorflow_int8"
        pred_q = forecaster.predict(test_data[0], quantize=True)
        eval_q = forecaster.evaluate(test_data, quantize=True)
        assert pred_q.shape == test_data[1].shape

if __name__ == '__main__':
    pytest.main([__file__])
