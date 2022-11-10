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

import os
import tempfile
import numpy as np
from unittest import TestCase
import pytest

from bigdl.chronos.utils import LazyImport
torch = LazyImport('torch')
NBeatsForecaster = LazyImport('bigdl.chronos.forecaster.nbeats_forecaster.NBeatsForecaster')
from .. import op_all, op_torch, op_distributed, op_onnxrt16, op_diff_set_all


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
    df = pd.DataFrame(np.random.rand(1000, 1),
                      columns=['value1'],
                      index=timeseries)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'timeseries'}, inplace=True)
    train, _, test = TSDataset.from_pandas(df=df,
                                           dt_col='timeseries',
                                           target_col=['value1'],
                                           with_split=True)
    if roll:
        for tsdata in [train, test]:
            tsdata.roll(lookback=24, horizon=horizon)
    return train, test


def create_tsdataset_val(roll=True, horizon=5):
    from bigdl.chronos.data import TSDataset
    import pandas as pd
    timeseries = pd.date_range(start='2020-01-01', freq='D', periods=1000)
    df = pd.DataFrame(np.random.rand(1000, 1),
                      columns=['value1'],
                      index=timeseries)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'timeseries'}, inplace=True)
    train, val, test = TSDataset.from_pandas(df=df,
                                             dt_col='timeseries',
                                             target_col=['value1'],
                                             with_split=True,
                                             val_ratio = 0.1)
    if roll:
        for tsdata in [train, test]:
            tsdata.roll(lookback=24, horizon=horizon)
    return train, val, test


@op_all
@op_torch
class TestChronosNBeatsForecaster(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_nbeats_forecaster_fit_pred_eva(self):
        train_data, _, test_data = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      stack_types=('generic', 'generic'),
                                      nb_blocks_per_stack=3,
                                      hidden_layer_units=256,
                                      metrics=['mae'],
                                      lr=0.01)
        forecaster.fit((train_data[0], train_data[1]), epochs=2)
        # inference
        nbeats_pred=forecaster.predict(test_data[0], acceleration=False)
        assert nbeats_pred.shape == test_data[1].shape
        eva = forecaster.evaluate(test_data, acceleration=False)
        assert eva[0].shape == test_data[1].shape[1:]

    @op_diff_set_all
    @op_onnxrt16
    def test_nbeats_forecaster_fit_loader(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      loss='mae',
                                      lr=0.01)
        forecaster.fit(train_loader, epochs=2)

        forecaster.quantize(calib_data=train_loader,
                    val_data=val_loader,
                    metric="mae",
                    framework=['onnxrt_qlinearops', 'pytorch_fx'])
        yhat = forecaster.predict(data=test_loader, acceleration=False)
        q_yhat = forecaster.predict(data=test_loader, quantize=True, acceleration=False)
        q_onnx_yhat = forecaster.predict_with_onnx(data=test_loader, quantize=True)
        assert yhat.shape == q_onnx_yhat.shape == q_yhat.shape == (400, 5, 1)
        forecaster.evaluate(test_loader, batch_size=32, acceleration=False)
        forecaster.evaluate_with_onnx(test_loader)
        forecaster.evaluate_with_onnx(test_loader, batch_size=32, quantize=True)

    @op_diff_set_all
    @op_onnxrt16
    def test_nbeats_forecaster_onnx_methods(self):
        train_data, val_data, test_data = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      loss='mae',
                                      lr=0.01)
        forecaster.fit(train_data, epochs=2)
        try:
            import onnx
            import onnxruntime
            pred = forecaster.predict(test_data[0], acceleration=False)
            pred_onnx = forecaster.predict_with_onnx(test_data[0])
            np.testing.assert_almost_equal(pred, pred_onnx, decimal=5)
            mse = forecaster.evaluate(test_data, multioutput='raw_values', acceleration=False)
            mse_onnx = forecaster.evaluate_with_onnx(test_data,
                                                     multioutput='raw_values')
            np.testing.assert_almost_equal(mse, mse_onnx, decimal=5)
            with pytest.raises(RuntimeError):
                forecaster.build_onnx(sess_options=1)
            forecaster.build_onnx(thread_num=1)
            mse = forecaster.evaluate(test_data, acceleration=False)
            mse_onnx = forecaster.evaluate_with_onnx(test_data)
            np.testing.assert_almost_equal(mse, mse_onnx, decimal=5)
        except ImportError:
            pass

    @op_diff_set_all
    def test_nbeats_forecaster_openvino_methods(self):
        train_data, val_data, test_data = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      loss='mae',
                                      lr=0.01)
        forecaster.fit(train_data, epochs=2)
        try:
            pred = forecaster.predict(test_data[0], acceleration=False)
            pred_openvino = forecaster.predict_with_openvino(test_data[0])
            np.testing.assert_almost_equal(pred, pred_openvino, decimal=5)
        except ImportError:
            pass

        # test exporting the openvino
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "fp32_openvino")
            ckpt_name_q = os.path.join(tmp_dir_name, "int_openvino")
            forecaster.export_openvino_file(dirname=ckpt_name, quantized_dirname=ckpt_name_q)

    @op_diff_set_all
    def test_nbeats_forecaster_jit_methods(self):
        train_data, val_data, test_data = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      loss='mae',
                                      lr=0.01)
        forecaster.fit(train_data, epochs=2)
        try:
            pred = forecaster.predict(test_data[0])
            pred_jit = forecaster.predict_with_jit(test_data[0])
            np.testing.assert_almost_equal(pred, pred_jit, decimal=5)
        except ImportError:
            pass

        # test exporting the jit
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "fp32_jit")
            forecaster.export_torchscript_file(dirname=ckpt_name)

    @op_diff_set_all
    def test_nbeats_forecaster_quantization(self):
        train_data, val_data, test_data = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      loss='mae',
                                      lr=0.01)
        forecaster.fit(train_data, epochs=2)
        # no tunning quantization
        forecaster.quantize(train_data)
        pred_q = forecaster.predict(test_data[0], quantize=True, acceleration=False)
        eval_q = forecaster.evaluate(test_data, quantize=True, acceleration=False)

    @op_diff_set_all
    def test_nbeats_forecaster_quantization_tuning(self):
        train_data, val_data, test_data = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      loss='mae',
                                      lr=0.01)
        forecaster.fit(train_data, epochs=2)
        # quantization with tunning
        forecaster.quantize(train_data)
        pred_q = forecaster.predict(test_data[0], quantize=True, acceleration=False)
        eval_q = forecaster.evaluate(test_data, quantize=True, acceleration=False)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            ckpt_name_q = os.path.join(tmp_dir_name, "ckpt.q")
            test_pred_save = forecaster.predict(test_data[0], acceleration=False)
            test_pred_save_q = forecaster.predict(test_data[0], quantize=True, acceleration=False)
            forecaster.save(ckpt_name, ckpt_name_q)
            forecaster.load(ckpt_name, ckpt_name_q)
            test_pred_load = forecaster.predict(test_data[0], acceleration=False)
            test_pred_load_q = forecaster.predict(test_data[0], quantize=True, acceleration=False)
        np.testing.assert_almost_equal(test_pred_save, test_pred_load)
        np.testing.assert_almost_equal(test_pred_save_q, test_pred_load_q)

    @op_diff_set_all
    @op_onnxrt16
    def test_nbeats_forecaster_quantization_onnx(self):
        train_data, val_data, test_data = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      loss='mae',
                                      lr=0.01)
        forecaster.fit(train_data, epochs=2)
        # no tunning quantization
        forecaster.quantize(train_data, framework=['onnxrt_qlinearops'])
        pred_q = forecaster.predict_with_onnx(test_data[0], quantize=True)
        eval_q = forecaster.evaluate_with_onnx(test_data, quantize=True)

    @op_diff_set_all
    @op_onnxrt16
    def test_nbeats_forecaster_quantization_onnx_tuning(self):
        train_data, val_data, test_data = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      loss='mae',
                                      lr=0.01)
        forecaster.fit(train_data, epochs=2)
        # quantization with tunning
        forecaster.quantize(train_data, val_data=val_data,
                            metric="mse", relative_drop=0.1, max_trials=3,
                            framework=['onnxrt_qlinearops'])
        pred_q = forecaster.predict_with_onnx(test_data[0], quantize=True)
        eval_q = forecaster.evaluate_with_onnx(test_data, quantize=True)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "fp32_onnx")
            ckpt_name_q = os.path.join(tmp_dir_name, "int_onnx")
            forecaster.export_onnx_file(dirname=ckpt_name, quantized_dirname=ckpt_name_q)

    def test_nbeats_forecaster_save_load(self):
        train_data, val_data, test_data = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      loss='mae',
                                      lr=0.01)
        forecaster.fit(train_data, epochs=2)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            test_pred_save = forecaster.predict(test_data[0], acceleration=False)
            forecaster.save(ckpt_name)
            forecaster.load(ckpt_name)
            test_pred_load = forecaster.predict(test_data[0], acceleration=False)
        np.testing.assert_almost_equal(test_pred_save, test_pred_load)

    def test_nbeats_forecaster_runtime_error(self):
        _, _, test_data = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      loss="mae",
                                      lr=0.01)
        with pytest.raises(RuntimeError):
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                ckpt_name = os.path.join(tmp_dir_name, "ckpt")
                forecaster.save(ckpt_name)
        with pytest.raises(RuntimeError):
            forecaster.predict(test_data[0], acceleration=False)
        with pytest.raises(RuntimeError):
            forecaster.evaluate(test_data, acceleration=False)

    @op_distributed
    def test_nbeats_forecaster_xshard_input(self):
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
            forecaster = NBeatsForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          stack_types=("generic", "seasonality"),
                                          loss='mae',
                                          lr=0.01,
                                          distributed=distributed)
            forecaster.fit(train_data, epochs=2)
            distributed_pred = forecaster.predict(test_data, acceleration=False)
            distributed_eval = forecaster.evaluate(val_data, acceleration=False)
        stop_orca_context()

    @op_distributed
    @op_diff_set_all
    @op_onnxrt16
    def test_nbeats_forecaster_distributed(self):
        train_data, val_data, test_data = create_data()
        _train_loader, _, _test_loader = create_data(loader=True)
        from bigdl.orca import init_orca_context, stop_orca_context
        init_orca_context(cores=4, memory="4g")
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      stack_types=('generic', 'seasonality'),
                                      loss="mae",
                                      lr=0.01,
                                      distributed=True)
        forecaster.fit(train_data, epochs=2)
        distributed_pred = forecaster.predict(test_data[0], acceleration=False)
        distributed_eval = forecaster.evaluate(val_data, acceleration=False)

        model = forecaster.get_model()
        assert isinstance(model, torch.nn.Module)

        forecaster.to_local()
        local_pred = forecaster.predict(test_data[0], acceleration=False)
        local_eval = forecaster.evaluate(val_data, acceleration=False)

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
            test_pred_save = forecaster.predict(test_data[0], acceleration=False)
            forecaster.save(ckpt_name)
            forecaster.load(ckpt_name)
            test_pred_load = forecaster.predict(test_data[0], acceleration=False)
        np.testing.assert_almost_equal(test_pred_save, test_pred_load)

        stop_orca_context()

    @op_distributed
    def test_nbeats_forecaster_dataloader_distributed(self):
        from bigdl.orca import init_orca_context, stop_orca_context
        train_data, _, _ = create_data(loader=True)
        init_orca_context(cores=4, memory="4g")
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      stack_types=("generic", "seasonality"),
                                      loss="mae",
                                      lr=0.01,
                                      distributed=True)
        forecaster.fit(train_data, epochs=2)

        with pytest.raises(RuntimeError):
            forecaster = NBeatsForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          stack_types=("generic", "generic"),
                                          loss="mae",
                                          lr=0.01,
                                          distributed=True)

        stop_orca_context()

    def test_nbeats_customized_loss_metric(self):
        from torchmetrics.functional import mean_squared_error
        train_data, _, _ = create_data(loader=True)
        _, _, test_data = create_data()
        loss = torch.nn.L1Loss()
        def customized_metric(y_true, y_pred):
            return mean_squared_error(torch.from_numpy(y_pred),
                                      torch.from_numpy(y_true)).numpy()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      stack_types=("generic", "seasonality"),
                                      loss=loss,
                                      metrics=[customized_metric],
                                      lr=0.01)
        forecaster.fit(train_data, epochs=2)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            test_pred_save = forecaster.predict(test_data[0], acceleration=False)
            forecaster.save(ckpt_name)
            forecaster.load(ckpt_name)
            test_pred_load = forecaster.predict(test_data[0], acceleration=False)
        np.testing.assert_almost_equal(test_pred_save, test_pred_load)

    def test_nbeats_forecaster_fit_val(self):
        train_data, val_data, _ = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      stack_types=('generic', 'generic'),
                                      nb_blocks_per_stack=3,
                                      hidden_layer_units=256,
                                      metrics=['mae'],
                                      lr=0.01)
        val_loss = forecaster.fit((train_data[0], train_data[1]), val_data, epochs=10)

    def test_nbeats_forecaster_fit_loader_val(self):
        train_loader, val_loader, _ = create_data(loader=True)
        forecater = NBeatsForecaster(past_seq_len=24,
                                     future_seq_len=5,
                                     loss='mae',
                                     lr=0.01)
        val_loss = forecater.fit(train_loader, val_loader, epochs=10)

    def test_forecaster_from_tsdataset(self):
        train, test = create_tsdataset()
        nbeats = NBeatsForecaster.from_tsdataset(train,
                                                 stack_types=("generic", "seasnoality"),
                                                 share_weights_in_stack=True,
                                                 hidden_layer_units=32)
        nbeats.fit(train,
                   epochs=2,
                   batch_size=32)
        yhat = nbeats.predict(test, batch_size=32, acceleration=False)
        test.roll(lookback=nbeats.data_config['past_seq_len'],
                  horizon=nbeats.data_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape

        del nbeats
        train, test = create_tsdataset(roll=False, horizon=[1, 3, 5])
        nbeats = NBeatsForecaster.from_tsdataset(train,
                                                 past_seq_len=24,
                                                 future_seq_len=2,
                                                 stack_types=("generic", "seasnoality"),
                                                 share_weights_in_stack=True,
                                                 hidden_layer_units=32)
        nbeats.fit(train,
                   epochs=2,
                   batch_size=32)
        yhat = nbeats.predict(test, batch_size=None, acceleration=False)
        test.roll(lookback=nbeats.data_config['past_seq_len'],
                  horizon=nbeats.data_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape

    @op_diff_set_all
    @op_onnxrt16
    def test_forecaster_from_tsdataset_data_loader_onnx(self):
        train, test = create_tsdataset(roll=False)
        loader = train.to_torch_data_loader(lookback=24,
                                            horizon=5)
        test_loader = test.to_torch_data_loader(lookback=24,
                                                horizon=5)
        nbeats = NBeatsForecaster.from_tsdataset(train)
        nbeats.fit(loader, epochs=2)
        yhat = nbeats.predict(test, acceleration=False)
        nbeats.quantize(calib_data=loader,
                        metric='mse',
                        framework=['pytorch_fx','onnxrt_qlinearops'])
        onnx_yhat = nbeats.predict_with_onnx(test)
        q_yhat = nbeats.predict(test, acceleration=False)
        q_onnx_yhat = nbeats.predict_with_onnx(test, quantize=True)
        assert onnx_yhat.shape == q_yhat.shape == yhat.shape == q_onnx_yhat.shape

        res = nbeats.evaluate(test_loader, acceleration=False)
        q_res = nbeats.evaluate(test_loader, quantize=True, acceleration=False)
        onnx_res = nbeats.evaluate_with_onnx(test_loader)
        q_onnx_res = nbeats.evaluate_with_onnx(test_loader, quantize=True)

    def test_nbeats_forecaster_fit_earlystop(self):
        train_data, val_data, _ = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      stack_types=('generic', 'generic'),
                                      nb_blocks_per_stack=3,
                                      hidden_layer_units=256,
                                      metrics=['mae'],
                                      lr=0.01)
        val_loss = forecaster.fit((train_data[0], train_data[1]), val_data,
                                  validation_mode='earlystop', epochs=10)

    def test_nbeats_forecaster_fit_earlystop_patience(self):
        train_data, val_data, _ = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      stack_types=('generic', 'generic'),
                                      nb_blocks_per_stack=3,
                                      hidden_layer_units=256,
                                      metrics=['mae'],
                                      lr=0.01)
        val_loss = forecaster.fit((train_data[0], train_data[1]), val_data,
                                  validation_mode='earlystop', earlystop_patience=6,
                                  epochs=10)

    def test_nbeats_forecaster_fit_best_val(self):
        train_data, val_data, _ = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      stack_types=('generic', 'generic'),
                                      nb_blocks_per_stack=3,
                                      hidden_layer_units=256,
                                      metrics=['mae'],
                                      lr=0.01)
        val_loss = forecaster.fit((train_data[0], train_data[1]), val_data,
                                  validation_mode='best_epoch', epochs=10)

    def test_predict_interval(self):
        train_data, val_data, test_data = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      stack_types=('generic', 'generic'),
                                      nb_blocks_per_stack=3,
                                      hidden_layer_units=256,
                                      metrics=['mse'],
                                      lr=0.01)
        forecaster.fit(train_data, epochs=2)
        y_pred, std = forecaster.predict_interval(data=test_data[0],
                                                  validation_data=val_data,
                                                  repetition_times=5)
        assert y_pred.shape == test_data[1].shape
        assert y_pred.shape == std.shape

    def test_forecaster_fit_val_from_tsdataset(self):
        train, val, test = create_tsdataset_val()
        nbeats = NBeatsForecaster.from_tsdataset(train,
                                                 stack_types=("generic", "seasnoality"),
                                                 share_weights_in_stack=True,
                                                 hidden_layer_units=32)
        nbeats.fit(train, val,
                   epochs=2,
                   batch_size=32)
        yhat = nbeats.predict(test, batch_size=32, acceleration=False)
        test.roll(lookback=nbeats.data_config['past_seq_len'],
                  horizon=nbeats.data_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape

        del nbeats
        train, val, test = create_tsdataset_val(roll=False, horizon=[1, 3, 5])
        nbeats = NBeatsForecaster.from_tsdataset(train,
                                                 past_seq_len=24,
                                                 future_seq_len=2,
                                                 stack_types=("generic", "seasnoality"),
                                                 share_weights_in_stack=True,
                                                 hidden_layer_units=32)
        nbeats.fit(train, val,
                   epochs=2,
                   batch_size=32)
        yhat = nbeats.predict(test, batch_size=None, acceleration=False)
        test.roll(lookback=nbeats.data_config['past_seq_len'],
                  horizon=nbeats.data_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape

    @op_diff_set_all
    @op_onnxrt16
    def test_forecaster_optimize_loader(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      stack_types=('generic', 'generic'),
                                      nb_blocks_per_stack=3,
                                      hidden_layer_units=256,
                                      metrics=['mse'],
                                      lr=0.01)
        forecaster.fit(train_loader, epochs=2)
        forecaster.optimize(train_data=train_loader,
                        validation_data=val_loader,
                        batch_size=32)
        forecaster.evaluate(val_loader)
        forecaster.predict(test_loader)

    def test_forecaster_predict_without_optimize(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      stack_types=('generic', 'generic'),
                                      nb_blocks_per_stack=3,
                                      hidden_layer_units=256,
                                      metrics=['mse'],
                                      lr=0.01)
        forecaster.fit(train_loader, epochs=2)
        forecaster.evaluate(val_loader)
        forecaster.predict(test_loader)
        assert forecaster.optim_model is None

    def test_nbeats_forecaster_eval_shuffle_loader(self):
        from torch.utils.data import DataLoader, TensorDataset
        from numpy.testing import assert_almost_equal
        train_data, val_data, test_data = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      loss='mse',
                                      lr=0.01)
        forecaster.fit(train_data, epochs=2)
        test_loader_shuffle_f = DataLoader(TensorDataset(torch.from_numpy(test_data[0]),
                                                         torch.from_numpy(test_data[1])),
                                           batch_size=32,
                                           shuffle=False)
        test_loader_shuffle_t = DataLoader(TensorDataset(torch.from_numpy(test_data[0]),
                                                         torch.from_numpy(test_data[1])),
                                           batch_size=32,
                                           shuffle=True)
        eval_f = forecaster.evaluate(test_loader_shuffle_f)
        eval_t = forecaster.evaluate(test_loader_shuffle_t)
        assert_almost_equal(eval_f, eval_t)

    @op_onnxrt16
    def test_nbeats_forecaster_eval_with_onnx_shuffle_loader(self):
        from torch.utils.data import DataLoader, TensorDataset
        from numpy.testing import assert_almost_equal
        train_data, val_data, test_data = create_data()
        forecaster = NBeatsForecaster(past_seq_len=24,
                                      future_seq_len=5,
                                      loss='mse',
                                      lr=0.01)
        forecaster.fit(train_data, epochs=2)
        test_loader_shuffle_f = DataLoader(TensorDataset(torch.from_numpy(test_data[0]),
                                                         torch.from_numpy(test_data[1])),
                                           batch_size=32,
                                           shuffle=False)
        test_loader_shuffle_t = DataLoader(TensorDataset(torch.from_numpy(test_data[0]),
                                                         torch.from_numpy(test_data[1])),
                                           batch_size=32,
                                           shuffle=True)
        eval_f = forecaster.evaluate_with_onnx(test_loader_shuffle_f)
        eval_t = forecaster.evaluate_with_onnx(test_loader_shuffle_t)
        assert_almost_equal(eval_f, eval_t)
