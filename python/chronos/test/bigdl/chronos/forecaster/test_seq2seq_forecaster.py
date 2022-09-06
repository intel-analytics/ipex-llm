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

import numpy as np
import tempfile
import os
import torch

from bigdl.chronos.forecaster.seq2seq_forecaster import Seq2SeqForecaster
from unittest import TestCase
import pytest

from .. import op_all, op_onnxrt16

def create_data(loader=False):
    num_train_samples = 1000
    num_val_samples = 400
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
    val_data = get_x_y(num_val_samples)
    test_data = get_x_y(num_test_samples)

    if loader:
        from torch.utils.data import DataLoader, TensorDataset
        train_loader = DataLoader(TensorDataset(torch.from_numpy(train_data[0]),
                                                torch.from_numpy(train_data[1])), batch_size=32)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(val_data[0]),
                                              torch.from_numpy(val_data[1])), batch_size=32)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(test_data[0]),
                                               torch.from_numpy(test_data[1])), batch_size=32)
        return train_loader, val_loader, test_loader
    else:
        return train_data, val_data, test_data


def create_tsdataset(roll=True, horizon=5):
    from bigdl.chronos.data import TSDataset
    import pandas as pd
    timeseries = pd.date_range(start='2020-01-01', freq='D', periods=1000)
    df = pd.DataFrame(np.random.rand(1000, 2),
                      columns=['value1', 'value2'],
                      index=timeseries)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'timeseries'}, inplace=True)
    train, _, test = TSDataset.from_pandas(df=df,
                                           dt_col='timeseries',
                                           target_col=['value1', 'value2'],
                                           with_split=True)
    if roll:
        for tsdata in [train, test]:
            tsdata.roll(lookback=24, horizon=horizon)
    return train, test


class TestChronosModelSeq2SeqForecaster(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_s2s_forecaster_fit_eva_pred(self):
        train_data, val_data, test_data = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01)
        forecaster.fit(train_data, epochs=2)
        test_pred = forecaster.predict(test_data[0])
        assert test_pred.shape == test_data[1].shape
        test_mse = forecaster.evaluate(test_data)
        assert test_mse[0].shape == test_data[1].shape[1:]

    @op_all
    @op_onnxrt16
    def test_s2s_forecaster_fit_loader(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01)
        train_loss = forecaster.fit(train_loader, epochs=2)
        yhat = forecaster.predict(data=test_loader)
        onnx_yhat = forecaster.predict_with_onnx(data=test_loader)
        assert yhat.shape == onnx_yhat.shape == (400, 5, 1)
        forecaster.evaluate(test_loader, batch_size=32)
        forecaster.evaluate_with_onnx(test_loader)
        forecaster.evaluate_with_onnx(test_loader, batch_size=32)

    @op_all
    @op_onnxrt16
    def test_s2s_forecaster_onnx_methods(self):
        train_data, val_data, test_data = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01)
        forecaster.fit(train_data, epochs=2)
        try:
            import onnx
            import onnxruntime
            pred = forecaster.predict(test_data[0])
            pred_onnx = forecaster.predict_with_onnx(test_data[0])
            np.testing.assert_almost_equal(pred, pred_onnx, decimal=5)
            mse = forecaster.evaluate(test_data, multioutput="raw_values")
            mse_onnx = forecaster.evaluate_with_onnx(test_data,
                                                     multioutput="raw_values")
            np.testing.assert_almost_equal(mse, mse_onnx, decimal=5)
            with pytest.raises(RuntimeError):
                forecaster.build_onnx(sess_options=1)
            forecaster.build_onnx(thread_num=1)
            mse = forecaster.evaluate(test_data)
            mse_onnx = forecaster.evaluate_with_onnx(test_data)
            np.testing.assert_almost_equal(mse, mse_onnx, decimal=5)
        except ImportError:
            pass

    def test_s2s_forecaster_openvino_methods(self):
        train_data, val_data, test_data = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01)
        forecaster.fit(train_data, epochs=2)
        try:
            pred = forecaster.predict(test_data[0])
            pred_openvino = forecaster.predict_with_openvino(test_data[0])
            np.testing.assert_almost_equal(pred, pred_openvino, decimal=5)
        except ImportError:
            pass

    def test_s2s_forecaster_quantization(self):
        train_data, val_data, test_data = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01)
        forecaster.fit(train_data, epochs=2)
        with pytest.raises(RuntimeError):
            forecaster.quantize(train_data)

    def test_s2s_forecaster_save_load(self):
        train_data, val_data, test_data = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01)
        forecaster.fit(train_data, epochs=2)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            test_pred_save = forecaster.predict(test_data[0])
            forecaster.save(ckpt_name)
            forecaster.load(ckpt_name)
            test_pred_load = forecaster.predict(test_data[0])
        np.testing.assert_almost_equal(test_pred_save, test_pred_load)

    def test_s2s_forecaster_runtime_error(self):
        train_data, val_data, test_data = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01)
        with pytest.raises(RuntimeError):
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                ckpt_name = os.path.join(tmp_dir_name, "ckpt")
                forecaster.save(ckpt_name)
        with pytest.raises(RuntimeError):
            forecaster.predict(test_data[0])
        with pytest.raises(RuntimeError):
            forecaster.evaluate(test_data)

    def test_s2s_forecaster_shape_error(self):
        train_data, val_data, test_data = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=2,
                                       loss="mae",
                                       lr=0.01)
        with pytest.raises(RuntimeError):
            forecaster.fit(train_data, epochs=2)

    def test_s2s_forecaster_xshard_input(self):
        from bigdl.orca import init_orca_context, stop_orca_context
        train_data, val_data, test_data = create_data()
        print("original", train_data[0].dtype)
        init_orca_context(cores=4, memory="2g")
        from bigdl.orca.data import XShards

        def transform_to_dict(data):
            return {'x': data[0], 'y': data[1]}

        def transform_to_dict_x(data):
            return {'x': data[0]}
        train_data = XShards.partition(train_data).transform_shard(transform_to_dict)
        val_data = XShards.partition(val_data).transform_shard(transform_to_dict)
        test_data = XShards.partition(test_data).transform_shard(transform_to_dict_x)
        for distributed in [True, False]:
            forecaster = Seq2SeqForecaster(past_seq_len=24,
                                           future_seq_len=5,
                                           input_feature_num=1,
                                           output_feature_num=1,
                                           loss="mae",
                                           lr=0.01,
                                           distributed=distributed)
            forecaster.fit(train_data, epochs=2)
            distributed_pred = forecaster.predict(test_data)
            distributed_eval = forecaster.evaluate(val_data)
        stop_orca_context()

    @op_all
    @op_onnxrt16
    def test_s2s_forecaster_distributed(self):
        from bigdl.orca import init_orca_context, stop_orca_context
        train_data, val_data, test_data = create_data()
        _train_loader, _, _test_loader = create_data(loader=True)

        init_orca_context(cores=4, memory="2g")

        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01,
                                       distributed=True)

        forecaster.fit(train_data, epochs=2)
        distributed_pred = forecaster.predict(test_data[0])
        distributed_eval = forecaster.evaluate(val_data)

        model = forecaster.get_model()
        assert isinstance(model, torch.nn.Module)

        forecaster.to_local()
        local_pred = forecaster.predict(test_data[0])
        local_eval = forecaster.evaluate(val_data)

        np.testing.assert_almost_equal(distributed_pred, local_pred, decimal=5)

        try:
            import onnx
            import onnxruntime
            local_pred_onnx = forecaster.predict_with_onnx(test_data[0])
            distributed_pred_onnx = forecaster.predict_with_onnx(_test_loader)
            local_eval_onnx = forecaster.evaluate_with_onnx(val_data)
            distributed_eval_onnx = forecaster.evaluate_with_onnx(_test_loader)
            np.testing.assert_almost_equal(distributed_pred, local_pred_onnx, decimal=5)
        except ImportError:
            pass

        model = forecaster.get_model()
        assert isinstance(model, torch.nn.Module)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "checkpoint.ckpt")
            test_pred_save = forecaster.predict(test_data[0])
            forecaster.save(ckpt_name)
            forecaster.load(ckpt_name)
            test_pred_load = forecaster.predict(test_data[0])
        np.testing.assert_almost_equal(test_pred_save, test_pred_load)

        stop_orca_context()

    def test_s2s_dataloader_distributed(self):
        from bigdl.orca import init_orca_context, stop_orca_context
        train_data, _, _ = create_data(loader=True)
        init_orca_context(cores=4, memory="2g")
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01,
                                       distributed=True)
        forecaster.fit(train_data, epochs=2)
        stop_orca_context()
    
    def test_s2s_customized_loss_metric(self):
        from torchmetrics.functional import mean_squared_error
        train_data, _, _ = create_data(loader=True)
        _, _, test_data = create_data()
        loss = torch.nn.L1Loss()
        def customized_metric(y_true, y_pred):
            return mean_squared_error(torch.from_numpy(y_pred),
                                      torch.from_numpy(y_true)).numpy()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss=loss,
                                       metrics=[customized_metric],
                                       lr=0.01)
        forecaster.fit(train_data, epochs=2)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            test_pred_save = forecaster.predict(test_data[0])
            forecaster.save(ckpt_name)
            forecaster.load(ckpt_name)
            test_pred_load = forecaster.predict(test_data[0])
        np.testing.assert_almost_equal(test_pred_save, test_pred_load)

    def test_s2s_forecaster_fit_val(self):
        train_data, val_data, _ = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01)
        val_loss = forecaster.fit(train_data, val_data, epochs=10)

    def test_s2s_forecaster_fit_loader_val(self):
        train_loader, val_loarder, _ = create_data(loader=True)
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01)
        val_loss = forecaster.fit(train_loader, val_loarder, epochs=10)

    def test_forecaster_from_tsdataset(self):
        train, test = create_tsdataset()
        s2s = Seq2SeqForecaster.from_tsdataset(train,
                                               lstm_hidden_dim=16,
                                               lstm_layer_num=1)
        s2s.fit(train,
                epochs=2,
                batch_size=32)
        yhat = s2s.predict(test, batch_size=32)
        test.roll(lookback=s2s.data_config['past_seq_len'],
                  horizon=s2s.data_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape

        del s2s
        train, test = create_tsdataset(roll=False, horizon=[1, 3, 5])
        s2s = Seq2SeqForecaster.from_tsdataset(train,
                                               past_seq_len=24,
                                               future_seq_len=2,
                                               lstm_hidden_dim=16,
                                               lstm_layer_num=1)
        s2s.fit(train,
                epochs=2,
                batch_size=32)
        yhat = s2s.predict(test, batch_size=None)
        test.roll(lookback=s2s.data_config['past_seq_len'],
                  horizon=s2s.data_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape

    @op_all
    @op_onnxrt16
    def test_forecaster_from_tsdataset_data_loader_onnx(self):
        train, test = create_tsdataset(roll=False)
        train.gen_dt_feature(one_hot_features=['WEEK'])
        test.gen_dt_feature(one_hot_features=['WEEK'])
        loader = train.to_torch_data_loader(lookback=24,
                                            horizon=5)
        test_loader = test.to_torch_data_loader(lookback=24,
                                                horizon=5)
        s2s = Seq2SeqForecaster.from_tsdataset(train)
        s2s.fit(loader, epochs=2)
        yhat = s2s.predict(test)
        onnx_yhat = s2s.predict_with_onnx(test)
        assert yhat.shape == onnx_yhat.shape

        res = s2s.evaluate(test_loader)
        onnx_res = s2s.evaluate_with_onnx(test_loader)

    def test_s2s_forecaster_fit_earlystop(self):
        train_data, val_data, _ = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01)
        val_loss = forecaster.fit(train_data, val_data,
                                  validation_mode='earlystop', epochs=10)

    def test_s2s_forecaster_fit_earlystop_patience(self):
        train_data, val_data, _ = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01)
        val_loss = forecaster.fit(train_data, val_data, validation_mode='earlystop',
                                  earlystop_patience=6, epochs=10)

    def test_s2s_forecaster_fit_best_val(self):
        train_data, val_data, _ = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       loss="mae",
                                       lr=0.01)
        val_loss = forecaster.fit(train_data, val_data,
                                  validation_mode='best_epoch', epochs=10)

    def test_s2s_forecaster_tune_fit(self):
        train_data, val_data, _ = create_data()
        import bigdl.nano.automl.hpo.space as space
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                        future_seq_len=5,
                                        input_feature_num=1,
                                        output_feature_num=1,
                                        loss="mae",
                                        lstm_hidden_dim=space.Categorical(32, 64, 128),
                                        lr=space.Real(0.001, 0.1, log=True))
        forecaster.tune(train_data, val_data, epochs=10,
                        n_trials=3, target_metric="mse",
                        direction="minimize")
        forecaster.fit(train_data, epochs=10)
