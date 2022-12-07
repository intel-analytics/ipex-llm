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

from bigdl.chronos.utils import LazyImport
torch = LazyImport('torch')
TCNForecaster = LazyImport('bigdl.chronos.forecaster.tcn_forecaster.TCNForecaster')
from unittest import TestCase
import pytest
from .. import op_torch, op_distributed, op_automl, op_inference, op_diff_set_all


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


def create_tsdataset(roll=True, horizon=5, val_ratio=0):
    from bigdl.chronos.data import TSDataset
    import pandas as pd
    timeseries = pd.date_range(start='2020-01-01', freq='D', periods=1000)
    df = pd.DataFrame(np.random.rand(1000, 2),
                      columns=['value1', 'value2'],
                      index=timeseries,
                      dtype=np.float32)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'timeseries'}, inplace=True)
    if val_ratio == 0:
        train, _, test = TSDataset.from_pandas(df=df,
                                            dt_col='timeseries',
                                            target_col=['value1', 'value2'],
                                            with_split=True)
        if roll:
            for tsdata in [train, test]:
                tsdata.roll(lookback=24, horizon=horizon)
        return train, test
    else:
        train, val, test = TSDataset.from_pandas(df=df,
                                            dt_col='timeseries',
                                            target_col=['value1', 'value2'],
                                            with_split=True,
                                            val_ratio=val_ratio)
        if roll:
            for tsdata in [train, val, test]:
                tsdata.roll(lookback=24, horizon=horizon)
        return train, val, test


def create_tsdataset_val(roll=True, horizon=5):
    from bigdl.chronos.data import TSDataset
    import pandas as pd
    timeseries = pd.date_range(start='2020-01-01', freq='D', periods=1000)
    df = pd.DataFrame(np.random.rand(1000, 2),
                      columns=['value1', 'value2'],
                      index=timeseries,
                      dtype=np.float32)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'timeseries'}, inplace=True)
    train, val, test = TSDataset.from_pandas(df=df,
                                             dt_col='timeseries',
                                             target_col=['value1', 'value2'],
                                             with_split=True,
                                             val_ratio = 0.1)
    if roll:
        for tsdata in [train, test]:
            tsdata.roll(lookback=24, horizon=horizon)
    return train, val, test


@op_torch
class TestChronosModelTCNForecaster(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tcn_forecaster_fit_eva_pred(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   loss="mae",
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        test_pred = forecaster.predict(test_data[0], acceleration=False)
        assert test_pred.shape == test_data[1].shape
        test_mse = forecaster.evaluate(test_data, acceleration=False)
        assert test_mse[0].shape == test_data[1].shape[1:]

    @op_diff_set_all
    @op_inference
    def test_tcn_forecaster_fit_loader(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   loss="mae",
                                   lr=0.01)
        train_loss = forecaster.fit(train_loader, epochs=2)
        forecaster.quantize(calib_data=train_loader,
                            val_data=val_loader,
                            metric="mae",
                            framework='pytorch_fx')
        q_yhat = forecaster.predict(data=test_loader, quantize=True, acceleration=False)
        yhat = forecaster.predict(data=test_loader, acceleration=False)
        forecaster.evaluate(test_loader, batch_size=32, acceleration=False)
        forecaster.quantize(calib_data=train_loader,
                            val_data=val_loader,
                            metric="mae",
                            framework='onnxrt_qlinearops')
        q_onnx_yhat = forecaster.predict_with_onnx(data=test_loader, quantize=True)
        forecaster.evaluate_with_onnx(test_loader, batch_size=32, quantize=True)
        forecaster.evaluate_with_onnx(test_loader)
        assert yhat.shape == q_onnx_yhat.shape == q_yhat.shape == (400, 5, 1)

    @op_diff_set_all
    @op_inference
    def test_tcn_forecaster_fit_loader_normalization_decomposation(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   normalization=True,
                                   decomposition_kernel_size=3,
                                   num_channels=[16, 16],
                                   loss="mae",
                                   lr=0.01)
        train_loss = forecaster.fit(train_loader, epochs=2)
        forecaster.quantize(calib_data=train_loader,
                            val_data=val_loader,
                            metric="mae",
                            framework='pytorch_fx')
        q_yhat = forecaster.predict(data=test_loader, quantize=True, acceleration=False)
        yhat = forecaster.predict(data=test_loader, acceleration=False)
        forecaster.evaluate(test_loader, batch_size=32, acceleration=False)
        forecaster.quantize(calib_data=train_loader,
                            val_data=val_loader,
                            metric="mae",
                            framework='onnxrt_qlinearops')
        q_onnx_yhat = forecaster.predict_with_onnx(data=test_loader, quantize=True)
        forecaster.evaluate_with_onnx(test_loader, batch_size=32, quantize=True)
        forecaster.evaluate_with_onnx(test_loader)
        assert yhat.shape == q_onnx_yhat.shape == q_yhat.shape == (400, 5, 1)

    @op_automl
    def test_tcn_forecaster_tune(self):
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, _ = create_data(loader=False)
        forecaster = TCNForecaster(past_seq_len=24,
                                future_seq_len=5,
                                input_feature_num=1,
                                output_feature_num=1,
                                kernel_size=4,
                                num_channels=[16, 16],
                                loss="mae",
                                metrics=['mae', 'mse', 'mape'],
                                lr=space.Real(0.001, 0.01, log=True))
        forecaster.tune(train_data, validation_data=val_data,
                        n_trials=2, target_metric='mse', direction="minimize")
        train_data = (train_data[0] * 10000.0, train_data[1] * 10000.0)
        forecaster.fit(train_data, epochs=2)
        train_loss = forecaster.trainer.callback_metrics['train/loss']
        assert train_loss > 10

    @op_automl
    def test_tcn_forecaster_tune_multi_processes(self):
        name = "parallel-example-torch"
        storage = "sqlite:///example_tcn.db"  # take sqlite for test, recommand to use mysql
        if os.path.exists("./example_tcn.db"):
            os.remove("./example_tcn.db")
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, _ = create_data(loader=False)
        forecaster = TCNForecaster(past_seq_len=24,
                                future_seq_len=5,
                                input_feature_num=1,
                                output_feature_num=1,
                                kernel_size=4,
                                num_channels=[16, 16],
                                loss="mae",
                                metrics=['mae', 'mse', 'mape'],
                                lr=space.Real(0.001, 0.01, log=True))
        forecaster.tune(train_data, validation_data=val_data,
                        n_trials=2, target_metric='mse', direction="minimize",
                        study_name=name,
                        storage=storage, n_parallels=2)
        forecaster.fit(train_data, epochs=2)
        os.remove("./example_tcn.db")

    @op_automl
    def test_tcn_forecaster_tune_multi_processes_normalization_decomposation(self):
        name = "parallel-example-torch"
        storage = "sqlite:///example_tcn.db"  # take sqlite for test, recommand to use mysql
        if os.path.exists("./example_tcn.db"):
            os.remove("./example_tcn.db")
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, _ = create_data(loader=False)
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   normalization=True,
                                   decomposition_kernel_size=3,
                                   num_channels=[16, 16],
                                   loss="mae",
                                   metrics=['mae', 'mse', 'mape'],
                                   lr=space.Real(0.001, 0.01, log=True))
        forecaster.tune(train_data, validation_data=val_data,
                        n_trials=2, target_metric='mse', direction="minimize",
                        study_name=name,
                        storage=storage, n_parallels=2)
        forecaster.fit(train_data, epochs=2)
        os.remove("./example_tcn.db")

    @op_automl
    def test_tcn_forecaster_multi_objective_tune(self):
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, _ = create_data(loader=False)
        forecaster = TCNForecaster(past_seq_len=24,
                                future_seq_len=5,
                                input_feature_num=1,
                                output_feature_num=1,
                                kernel_size=4,
                                num_channels=[16, 16],
                                loss="mae",
                                metrics=['mae', 'mse', 'mape'],
                                lr=space.Real(0.001, 0.01, log=True))
        forecaster.num_processes = 1
        forecaster.tune(train_data, validation_data=val_data,
                        n_trials=2, target_metric=['mse', 'latency'],
                        direction=None,
                        directions=["minimize", "minimize"])

    @op_automl
    @op_diff_set_all
    @op_inference
    def test_tcn_forecaster_multi_objective_tune_acceleration(self):
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, _ = create_data(loader=False)
        forecaster = TCNForecaster(past_seq_len=24,
                                future_seq_len=5,
                                input_feature_num=1,
                                output_feature_num=1,
                                kernel_size=4,
                                num_channels=[16, 16],
                                loss="mae",
                                metrics=['mae', 'mse', 'mape'],
                                lr=space.Real(0.001, 0.01, log=True))
        forecaster.num_processes = 1
        forecaster.tune(train_data, validation_data=val_data,
                        n_trials=2, target_metric=['mse', 'latency'], 
                        directions=["minimize", "minimize"],
                        acceleration=True, direction=None)

    @op_automl
    @op_inference
    def test_tcn_forecaster_mo_tune_acceleration_fit_input(self):
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, _ = create_data(loader=False)
        forecaster = TCNForecaster(past_seq_len=24,
                                future_seq_len=5,
                                input_feature_num=1,
                                output_feature_num=1,
                                kernel_size=4,
                                num_channels=[16, 16],
                                loss="mae",
                                metrics=['mae', 'mse', 'mape'],
                                lr=space.Real(0.001, 0.01, log=True))
        forecaster.num_processes = 1
        forecaster.tune(train_data, validation_data=val_data,
                        n_trials=2, target_metric=['mse', 'latency'], 
                        directions=["minimize", "minimize"],
                        acceleration=True, direction=None)
        with self.assertRaises(Exception):
            forecaster.fit(train_data, epochs=2)

    @op_automl
    @op_inference
    def test_tcn_forecaster_mo_tune_acceleration_fit(self):
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, _ = create_data(loader=False)
        forecaster = TCNForecaster(past_seq_len=24,
                                future_seq_len=5,
                                input_feature_num=1,
                                output_feature_num=1,
                                kernel_size=4,
                                num_channels=[16, 16],
                                loss="mae",
                                metrics=['mae', 'mse', 'mape'],
                                lr=space.Real(0.001, 0.01, log=True))
        forecaster.num_processes = 1
        forecaster.tune(train_data, validation_data=val_data,
                        n_trials=2, target_metric=['mse', 'latency'], 
                        directions=["minimize", "minimize"],
                        acceleration=True, direction=None)
        forecaster.fit(train_data, epochs=2, use_trial_id=0)

    @op_diff_set_all
    @op_inference
    def test_tcn_forecaster_onnx_methods(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        try:
            import onnx
            import onnxruntime
            pred = forecaster.predict(test_data[0], acceleration=False)
            pred_onnx = forecaster.predict_with_onnx(test_data[0])
            np.testing.assert_almost_equal(pred, pred_onnx, decimal=5)
            mse = forecaster.evaluate(test_data, multioutput="raw_values", acceleration=False)
            mse_onnx = forecaster.evaluate_with_onnx(test_data,
                                                     multioutput="raw_values")
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
    @op_inference
    def test_tcn_forecaster_onnx_methods_normalization_decomposation(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   normalization=True,
                                   decomposition_kernel_size=3,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        try:
            import onnx
            import onnxruntime
            pred = forecaster.predict(test_data[0], acceleration=False)
            pred_onnx = forecaster.predict_with_onnx(test_data[0])
            np.testing.assert_almost_equal(pred, pred_onnx, decimal=5)
            mse = forecaster.evaluate(test_data, multioutput="raw_values", acceleration=False)
            mse_onnx = forecaster.evaluate_with_onnx(test_data,
                                                     multioutput="raw_values")
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
    def test_tcn_forecaster_openvino_methods(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        try:
            pred = forecaster.predict(test_data[0], acceleration=False)
            pred_openvino = forecaster.predict_with_openvino(test_data[0])
            np.testing.assert_almost_equal(pred, pred_openvino, decimal=5)
        except ImportError:
            pass

        forecaster.quantize(calib_data=train_data,
                            framework="openvino")
        q_openvino_yhat = forecaster.predict_with_openvino(test_data[0], quantize=True)
        openvino_yhat = forecaster.predict_with_openvino(test_data[0])
        assert openvino_yhat.shape == q_openvino_yhat.shape == test_data[1].shape

        # test exporting the openvino
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "fp32_openvino")
            ckpt_name_q = os.path.join(tmp_dir_name, "int_openvino")
            forecaster.export_openvino_file(dirname=ckpt_name, quantized_dirname=ckpt_name_q)

    @op_diff_set_all
    def test_tcn_forecaster_openvino_methods_normalization_decomposation(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   normalization=True,
                                   decomposition_kernel_size=3,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        try:
            pred = forecaster.predict(test_data[0], acceleration=False)
            pred_openvino = forecaster.predict_with_openvino(test_data[0])
            np.testing.assert_almost_equal(pred, pred_openvino, decimal=5)
        except ImportError:
            pass

        forecaster.quantize(calib_data=train_data,
                            framework="openvino")
        q_openvino_yhat = forecaster.predict_with_openvino(test_data[0], quantize=True)
        openvino_yhat = forecaster.predict_with_openvino(test_data[0])
        assert openvino_yhat.shape == q_openvino_yhat.shape == test_data[1].shape

        # test exporting the openvino
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "fp32_openvino")
            ckpt_name_q = os.path.join(tmp_dir_name, "int_openvino")
            forecaster.export_openvino_file(dirname=ckpt_name, quantized_dirname=ckpt_name_q)

    @op_inference
    def test_tcn_forecaster_openvino_methods_loader(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train_loader, epochs=2)
        try:
            pred = forecaster.predict(test_loader, acceleration=False)
            pred_openvino = forecaster.predict_with_openvino(test_loader)
            np.testing.assert_almost_equal(pred, pred_openvino, decimal=5)
        except ImportError:
            pass

        forecaster.quantize(calib_data=train_loader,
                            framework="openvino")
        q_openvino_yhat = forecaster.predict_with_openvino(test_loader, quantize=True)
        openvino_yhat = forecaster.predict_with_openvino(test_loader)
        assert openvino_yhat.shape == q_openvino_yhat.shape

    @op_inference
    def test_tcn_forecaster_openvino_methods_tsdataset(self):
        train, test = create_tsdataset(roll=True, horizon=5)
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=2,
                                   output_feature_num=2,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train, epochs=2)
        try:
            pred = forecaster.predict(test, acceleration=False)
            pred_openvino = forecaster.predict_with_openvino(test)
            np.testing.assert_almost_equal(pred, pred_openvino, decimal=5)
        except ImportError:
            pass

        forecaster.quantize(calib_data=train,
                            framework="openvino")
        q_openvino_yhat = forecaster.predict_with_openvino(test, quantize=True)
        openvino_yhat = forecaster.predict_with_openvino(test)
        assert openvino_yhat.shape == q_openvino_yhat.shape

    @op_diff_set_all
    def test_tcn_forecaster_jit_methods(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
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
    def test_tcn_forecaster_jit_methods_normalization_decomposation(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   normalization=True,
                                   decomposition_kernel_size=3,
                                   num_channels=[16, 16],
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
    def test_tcn_forecaster_jit_methods_loader(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train_loader, epochs=2)
        try:
            pred = forecaster.predict(test_loader)
            pred_jit = forecaster.predict_with_jit(test_loader)
            np.testing.assert_almost_equal(pred, pred_jit, decimal=5)
        except ImportError:
            pass

    @op_diff_set_all
    def test_tcn_forecaster_jit_methods_tsdataset(self):
        train, test = create_tsdataset(roll=True, horizon=5)
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=2,
                                   output_feature_num=2,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train, epochs=2)
        try:
            pred = forecaster.predict(test)
            pred_jit = forecaster.predict_with_jit(test)
            np.testing.assert_almost_equal(pred, pred_jit, decimal=5)
        except ImportError:
            pass

    @op_diff_set_all
    def test_tcn_forecaster_quantization_dynamic(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        forecaster.quantize(approach="dynamic")
        pred_q = forecaster.predict(test_data[0], quantize=True, acceleration=False)
        eval_q = forecaster.evaluate(test_data, quantize=True, acceleration=False)
        # reset model
        forecaster.internal.train()
        forecaster.internal.eval()
        # dynamic quantization does not need calib data
        with pytest.raises(RuntimeError):
            forecaster.quantize(train_data, approach="dynamic")

    @op_diff_set_all
    def test_tcn_forecaster_quantization(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        # no tunning quantization
        forecaster.quantize(train_data)
        pred_q = forecaster.predict(test_data[0], quantize=True, acceleration=False)
        eval_q = forecaster.evaluate(test_data, quantize=True, acceleration=False)
        # reset model
        forecaster.internal.train()
        forecaster.internal.eval()
        # quantization with tunning
        forecaster.quantize(train_data, val_data=val_data,
                            metric="rmse", relative_drop=0.1, max_trials=3)
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
    def test_tcn_forecaster_quantization_normalization_decomposation(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   normalization=True,
                                   decomposition_kernel_size=3,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        # no tunning quantization
        forecaster.quantize(train_data)
        pred_q = forecaster.predict(test_data[0], quantize=True, acceleration=False)
        eval_q = forecaster.evaluate(test_data, quantize=True, acceleration=False)
        # reset model
        forecaster.internal.train()
        forecaster.internal.eval()
        # quantization with tunning
        forecaster.quantize(train_data, val_data=val_data,
                            metric="rmse", relative_drop=0.1, max_trials=3)
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
    @op_inference
    def test_tcn_forecaster_quantization_onnx(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        # no tunning quantization
        forecaster.quantize(train_data, framework='onnxrt_qlinearops')
        pred_q = forecaster.predict_with_onnx(test_data[0], quantize=True)
        eval_q = forecaster.evaluate_with_onnx(test_data, quantize=True)

    @op_diff_set_all
    @op_inference
    def test_tcn_forecaster_quantization_onnx_tuning(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        # quantization with tunning
        forecaster.quantize(train_data, val_data=val_data,
                            metric="mse", relative_drop=0.1, max_trials=3,
                            framework='onnxrt_qlinearops')
        pred_q = forecaster.predict_with_onnx(test_data[0], quantize=True)
        eval_q = forecaster.evaluate_with_onnx(test_data, quantize=True)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "fp32_onnx")
            ckpt_name_q = os.path.join(tmp_dir_name, "int_onnx")
            forecaster.export_onnx_file(dirname=ckpt_name, quantized_dirname=ckpt_name_q)

    def test_tcn_forecaster_save_load(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            test_pred_save = forecaster.predict(test_data[0], acceleration=False)
            forecaster.save(ckpt_name)
            forecaster.load(ckpt_name)
            test_pred_load = forecaster.predict(test_data[0], acceleration=False)
        np.testing.assert_almost_equal(test_pred_save, test_pred_load)

    def test_tcn_forecaster_runtime_error(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=3,
                                   lr=0.01)
        with pytest.raises(RuntimeError):
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                ckpt_name = os.path.join(tmp_dir_name, "ckpt")
                forecaster.save(ckpt_name)
        with pytest.raises(RuntimeError):
            forecaster.predict(test_data[0], acceleration=False)
        with pytest.raises(RuntimeError):
            forecaster.evaluate(test_data, acceleration=False)

    def test_tcn_forecaster_shape_error(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=2,
                                   kernel_size=3,
                                   lr=0.01)
        with pytest.raises(RuntimeError):
            forecaster.fit(train_data, epochs=2)

    @op_distributed
    def test_tcn_forecaster_xshard_input(self):
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
            forecaster = TCNForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       kernel_size=3,
                                       lr=0.01,
                                       distributed=distributed)
            forecaster.fit(train_data, epochs=2)
            distributed_pred = forecaster.predict(test_data, acceleration=False)
            distributed_eval = forecaster.evaluate(val_data, acceleration=False)
        stop_orca_context()

    @op_distributed
    def test_tcn_forecaster_xshard_input_of_validation_data(self):
        from bigdl.orca import init_orca_context, stop_orca_context
        train_data, val_data, test_data = create_data()
        init_orca_context(cores=4, memory="2g")
        from bigdl.orca.data import XShards

        def transform_to_dict(data):
            return {'x': data[0], 'y': data[1]}

        def transform_to_dict_x(data):
            return {'x': data[0]}
        train_data = XShards.partition(train_data).transform_shard(transform_to_dict)
        val_data = XShards.partition(val_data).transform_shard(transform_to_dict)
        test_data = XShards.partition(test_data).transform_shard(transform_to_dict_x)
        forecaster = TCNForecaster(past_seq_len=24,
                                    future_seq_len=5,
                                    input_feature_num=1,
                                    output_feature_num=1,
                                    kernel_size=3,
                                    lr=0.01,
                                    distributed=False)
        forecaster.fit(train_data, val_data, epochs=2)
        stop_orca_context()

    @op_distributed
    @op_diff_set_all
    @op_inference
    def test_tcn_forecaster_distributed(self):
        from bigdl.orca import init_orca_context, stop_orca_context
        train_data, val_data, test_data = create_data()
        _train_loader, _, _test_loader = create_data(loader=True)

        init_orca_context(cores=4, memory="2g")

        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=3,
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
    @op_diff_set_all
    @op_inference
    def test_tcn_forecaster_distributed_normalization_decomposation(self):
        from bigdl.orca import init_orca_context, stop_orca_context
        train_data, val_data, test_data = create_data()
        _train_loader, _, _test_loader = create_data(loader=True)

        init_orca_context(cores=4, memory="2g")

        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=3,
                                   normalization=True,
                                   decomposition_kernel_size=3,
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
    def test_tcn_dataloader_distributed(self):
        from bigdl.orca import init_orca_context, stop_orca_context
        train_loader, _, _ = create_data(loader=True)
        init_orca_context(cores=4, memory="2g")
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=3,
                                   lr=0.01,
                                   distributed=True)
        forecaster.fit(train_loader, epochs=2)
        stop_orca_context()

    def test_tcn_customized_loss_metric(self):
        from torchmetrics.functional import mean_squared_error
        train_loader, _, _ = create_data(loader=True)
        _, _, test_data = create_data()
        loss = torch.nn.L1Loss()
        def customized_metric(y_true, y_pred):
            return mean_squared_error(torch.from_numpy(y_pred),
                                      torch.from_numpy(y_true)).numpy()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=3,
                                   loss=loss,
                                   metrics=[customized_metric],
                                   lr=0.01)
        forecaster.fit(train_loader, epochs=2)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            test_pred_save = forecaster.predict(test_data[0], acceleration=False)
            forecaster.save(ckpt_name)
            forecaster.load(ckpt_name)
            test_pred_load = forecaster.predict(test_data[0], acceleration=False)
        np.testing.assert_almost_equal(test_pred_save, test_pred_load)
   
    def test_tcn_forecaster_fit_val(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   loss="mae",
                                   lr=0.01)
        val_loss = forecaster.fit(train_data, val_data, epochs=10)
        _ = forecaster.predict(test_data[0], acceleration=False)

    def test_tcn_forecaster_fit_loader_val(self):
        train_loader, val_loader, _ = create_data(loader=True)
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   loss="mae",
                                   lr=0.01)
        val_loss = forecaster.fit(train_loader, val_loader, epochs=10)

    def test_forecaster_from_tsdataset(self):
        train, test = create_tsdataset()
        tcn = TCNForecaster.from_tsdataset(train,
                                           num_channels=[16]*3)
        tcn.fit(train,
                epochs=2,
                batch_size=32)
        yhat = tcn.predict(test, batch_size=32, acceleration=False)
        test.roll(lookback=tcn.data_config['past_seq_len'],
                  horizon=tcn.data_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape

        del tcn
        train, test = create_tsdataset(roll=False, horizon=[1, 3, 5])
        tcn = TCNForecaster.from_tsdataset(train,
                                           past_seq_len=24,
                                           future_seq_len=5,
                                           num_channels=[16]*3)
        tcn.fit(train,
                epochs=2,
                batch_size=32)
        yhat = tcn.predict(test, batch_size=None, acceleration=False)
        test.roll(lookback=tcn.data_config['past_seq_len'],
                  horizon=tcn.data_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape

    @op_diff_set_all
    @op_inference
    def test_forecaster_from_tsdataset_data_loader_onnx(self):
        train, test = create_tsdataset(roll=False)
        train.gen_dt_feature(one_hot_features=['WEEK'])
        test.gen_dt_feature(one_hot_features=['WEEK'])

        loader = train.to_torch_data_loader(lookback=24,
                                            horizon=5)
        test_loader = test.to_torch_data_loader(lookback=24,
                                                horizon=5)
        tcn = TCNForecaster.from_tsdataset(train)

        tcn.fit(loader, epochs=2)
        yhat = tcn.predict(test, acceleration=False)
        res = tcn.evaluate(test_loader, acceleration=False)
        tcn.quantize(calib_data=loader,
                     metric='mse',
                     framework='pytorch_fx')
        q_yhat = tcn.predict(test, quantize=True, acceleration=False)
        q_res = tcn.evaluate(test_loader, quantize=True, acceleration=False)
        tcn.quantize(calib_data=loader,
                     metric='mse',
                     framework='onnxrt_qlinearops')
        q_onnx_yhat = tcn.predict_with_onnx(test, quantize=True)
        q_onnx_res = tcn.evaluate_with_onnx(test_loader, quantize=True)
        onnx_yhat = tcn.predict_with_onnx(test)
        onnx_res = tcn.evaluate_with_onnx(test_loader)
        assert onnx_yhat.shape == q_yhat.shape == yhat.shape == q_onnx_yhat.shape

    def test_tcn_forecaster_fit_earlystop(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   loss="mae",
                                   lr=0.01)
        train_loss = forecaster.fit(train_data, val_data, validation_mode='earlystop', epochs=10)

    def test_tcn_forecaster_fit_earlystop_patience(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   loss="mae",
                                   lr=0.01)
        train_loss = forecaster.fit(train_data, val_data, validation_mode='earlystop',
                                    earlystop_patience=6, epochs=10)

    def test_tcn_forecaster_fit_best_val(self):
        train_data, val_data, _ = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   loss="mae",
                                   lr=0.01)
        val_loss = forecaster.fit(train_data, val_data, validation_mode='best_epoch', epochs=10)

    @op_automl
    def test_tcn_forecaster_tune_save_load(self):
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=space.Categorical([16, 16], [4, 4]),
                                   lr=0.01)
        forecaster.tune(train_data, validation_data=val_data,
                        n_trials=2, target_metric='mse', direction="minimize")
        forecaster.fit(train_data, epochs=2)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            test_pred_save = forecaster.predict(test_data[0], acceleration=False)
            forecaster.save(ckpt_name)
            forecaster.load(ckpt_name)
            test_pred_load = forecaster.predict(test_data[0], acceleration=False)
        np.testing.assert_almost_equal(test_pred_save, test_pred_load)

    def test_predict_interval_with_numpy(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16, 16],
                                   loss="mse",
                                   metrics=["mse"],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        # only the first time needs validation_data
        y_pred, std = forecaster.predict_interval(data=test_data[0],
                                                  validation_data=val_data,
                                                  repetition_times=5)
        assert y_pred.shape == test_data[1].shape
        assert y_pred.shape == std.shape
        y_pred, std = forecaster.predict_interval(data=test_data[0])
    
    def test_predict_interval_with_loader(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16, 16],
                                   loss="mse",
                                   metrics=["mse"],
                                   lr=0.01)
        forecaster.fit(train_loader, epochs=2)
        # only the first time needs validation_data
        y_pred, std = forecaster.predict_interval(data=test_loader,
                                                  validation_data=val_loader,
                                                  repetition_times=5)
        assert y_pred.shape == std.shape
        y_pred, std = forecaster.predict_interval(data=test_loader)

    def test_predict_interval_with_tsdataset(self):
        train, val, test = create_tsdataset(roll=True, horizon=5, val_ratio=0.1)
        forecaster = TCNForecaster.from_tsdataset(train,
                                           num_channels=[16]*3)
        forecaster.fit(train, epochs=2, batch_size=32)
        # only the first time needs validation_data
        y_pred, std = forecaster.predict_interval(data=test,
                                                  validation_data=val,
                                                  repetition_times=5)
        assert y_pred.shape == std.shape
        y_pred, std = forecaster.predict_interval(data=test)

    def test_predict_interval_without_validation_data(self):
        train_data, _, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16, 16],
                                   loss="mse",
                                   metrics=["mse"],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        with pytest.raises(RuntimeError):
            y_pred, std = forecaster.predict_interval(data=test_data[0],
                                                      repetition_times=5)
    def test_forecaster_fit_val_from_tsdataset(self):
        train, val, test = create_tsdataset(val_ratio=0.1)
        tcn = TCNForecaster.from_tsdataset(train,
                                           num_channels=[16]*3)
        tcn.fit(train, val,
                epochs=2,
                batch_size=32)
        yhat = tcn.predict(test, batch_size=32, acceleration=False)
        test.roll(lookback=tcn.data_config['past_seq_len'],
                  horizon=tcn.data_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape

        del tcn
        train, val, test = create_tsdataset(roll=False, horizon=[1, 3, 5],
                                            val_ratio=0.1)
        tcn = TCNForecaster.from_tsdataset(train,
                                           past_seq_len=24,
                                           future_seq_len=5,
                                           num_channels=[16]*3)
        tcn.fit(train, val,
                epochs=2,
                batch_size=32)
        yhat = tcn.predict(test, batch_size=None, acceleration=False)
        test.roll(lookback=tcn.data_config['past_seq_len'],
                  horizon=tcn.data_config['future_seq_len'])
        _, y_test = test.to_numpy()
        assert yhat.shape == y_test.shape

    @op_diff_set_all
    @op_inference
    def test_forecaster_optimize_loader(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16, 16],
                                   loss="mse",
                                   metrics=["mse"],
                                   lr=0.01)
        forecaster.fit(train_loader, epochs=2)
        forecaster.optimize(train_data=train_loader,
                            validation_data=val_loader,
                            batch_size=32)
        forecaster.evaluate(val_loader)
        forecaster.predict(test_loader)

    @op_diff_set_all
    @op_inference
    def test_forecaster_optimize_numpy(self):
        train_data, val_data, test_data = create_data(loader=False)
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16, 16],
                                   loss="mse",
                                   metrics=["mse"],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        forecaster.optimize(train_data=train_data,
                            validation_data=val_data,
                            batch_size=32)
        forecaster.evaluate(val_data)
        forecaster.predict(test_data[0])

    @op_diff_set_all
    @op_inference
    def test_forecaster_optimize_tsdataset(self):
        train, val, test = create_tsdataset(roll=True, horizon=5, val_ratio=0.1)
        forecaster = TCNForecaster.from_tsdataset(train,
                                                  num_channels=[16]*3)
        forecaster.fit(train, epochs=2)
        forecaster.optimize(train_data=train,
                            validation_data=val,
                            batch_size=32)
        forecaster.evaluate(val)
        forecaster.predict(test)

    def test_forecaster_predict_without_optimize(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16, 16],
                                   loss="mse",
                                   metrics=["mse"],
                                   lr=0.01)
        forecaster.fit(train_loader, epochs=2)
        forecaster.evaluate(val_loader)
        forecaster.predict(test_loader)
        assert forecaster.accelerated_model is None

    @op_diff_set_all
    @op_inference
    def test_forecaster_optimize_loader_without_validation_data(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16, 16],
                                   loss="mse",
                                   metrics=["mse"],
                                   lr=0.01)
        forecaster.fit(train_loader, epochs=2)
        forecaster.optimize(train_data=train_loader,
                            batch_size=32)
        forecaster.evaluate(val_loader)
        forecaster.predict(test_loader)

    def test_tcn_forecaster_eval_shuffle_loader(self):
        from torch.utils.data import DataLoader, TensorDataset
        from numpy.testing import assert_almost_equal
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   loss="mse",
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

    def test_tcn_forecaster_dummy_encoder(self):
        from torch.utils.data import DataLoader, TensorDataset
        from numpy.testing import assert_almost_equal
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   dummy_encoder=True,
                                   loss="mse",
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

    @op_inference
    def test_tcn_forecaster_eval_with_onnx_shuffle_loader(self):
        from torch.utils.data import DataLoader, TensorDataset
        from numpy.testing import assert_almost_equal
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   loss="mse",
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
