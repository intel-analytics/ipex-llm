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

from bigdl.chronos.forecaster.lstm_forecaster import LSTMForecaster
from unittest import TestCase
import pytest


def create_data(loader=False):
    num_train_samples = 1000
    num_val_samples = 400
    num_test_samples = 400
    input_time_steps = 24
    input_feature_dim = 2
    output_time_steps = 1
    output_feature_dim = 2

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


class TestChronosModelLSTMForecaster(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_lstm_forecaster_fit_eva_pred(self):
        train_data, val_data, test_data = create_data()
        forecaster = LSTMForecaster(past_seq_len=24,
                                    input_feature_num=2,
                                    output_feature_num=2,
                                    hidden_dim=[32, 16],
                                    layer_num=2,
                                    dropout=[0.1, 0.2],
                                    loss="mae",
                                    lr=0.01)
        train_loss = forecaster.fit(train_data, epochs=2)
        test_pred = forecaster.predict(test_data[0])
        assert test_pred.shape == test_data[1].shape
        test_mse = forecaster.evaluate(test_data)
        assert test_mse[0].shape == test_data[1].shape[1:]

    def test_lstm_forecaster_fit_loader(self):
        train_loader, _, _ = create_data(loader=True)
        forecaster = LSTMForecaster(past_seq_len=24,
                                    input_feature_num=2,
                                    output_feature_num=2,
                                    hidden_dim=[32, 16],
                                    layer_num=2,
                                    dropout=[0.1, 0.2],
                                    loss="mae",
                                    lr=0.01)
        train_loss = forecaster.fit(train_loader, epochs=2)

    def test_lstm_forecaster_onnx_methods(self):
        train_data, val_data, test_data = create_data()
        forecaster = LSTMForecaster(past_seq_len=24,
                                    input_feature_num=2,
                                    output_feature_num=2,
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
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = 1
            forecaster.build_onnx(sess_options=sess_options)
            mse = forecaster.evaluate(test_data)
            mse_onnx = forecaster.evaluate_with_onnx(test_data)
            np.testing.assert_almost_equal(mse, mse_onnx, decimal=5)
        except ImportError:
            pass

    def test_lstm_forecaster_openvino_methods(self):
        train_data, val_data, test_data = create_data()
        forecaster = LSTMForecaster(past_seq_len=24,
                                    input_feature_num=2,
                                    output_feature_num=2,
                                    loss="mae",
                                    lr=0.01)
        forecaster.fit(train_data, epochs=2)
        try:
            pred = forecaster.predict(test_data[0])
            pred_openvino = forecaster.predict_with_openvino(test_data[0])
            np.testing.assert_almost_equal(pred, pred_openvino, decimal=5)
        except ImportError:
            pass

    def test_lstm_forecaster_quantization(self):
        train_data, val_data, test_data = create_data()
        forecaster = LSTMForecaster(past_seq_len=24,
                                    input_feature_num=2,
                                    output_feature_num=2,
                                    loss="mae",
                                    lr=0.01)
        forecaster.fit(train_data, epochs=2)
        # no tunning quantization
        forecaster.quantize(train_data)
        pred_q = forecaster.predict(test_data[0], quantize=True)
        eval_q = forecaster.evaluate(test_data, quantize=True)
        # reset model
        forecaster.internal.train()
        forecaster.internal.eval()
        # quantization with tunning
        forecaster.quantize(train_data, val_data=val_data,
                            metric="mse", relative_drop=0.1, max_trials=3)
        pred_q = forecaster.predict(test_data[0], quantize=True)
        eval_q = forecaster.evaluate(test_data, quantize=True)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            ckpt_name_q = os.path.join(tmp_dir_name, "ckpt.q")
            test_pred_save = forecaster.predict(test_data[0])
            test_pred_save_q = forecaster.predict(test_data[0], quantize=True)
            forecaster.save(ckpt_name, ckpt_name_q)
            forecaster.load(ckpt_name, ckpt_name_q)
            test_pred_load = forecaster.predict(test_data[0])
            test_pred_load_q = forecaster.predict(test_data[0], quantize=True)
        np.testing.assert_almost_equal(test_pred_save, test_pred_load)
        np.testing.assert_almost_equal(test_pred_save_q, test_pred_load_q)

    def test_lstm_forecaster_quantization_onnx(self):
        train_data, val_data, test_data = create_data()
        forecaster = LSTMForecaster(past_seq_len=24,
                                    input_feature_num=2,
                                    output_feature_num=2,
                                    loss="mae",
                                    lr=0.01)
        forecaster.fit(train_data, epochs=2)
        # no tunning quantization
        forecaster.quantize(train_data, framework=['onnxrt_qlinearops'])
        pred_q = forecaster.predict_with_onnx(test_data[0], quantize=True)
        eval_q = forecaster.evaluate_with_onnx(test_data, quantize=True)
        
    def test_lstm_forecaster_quantization_onnx_tuning(self):
        train_data, val_data, test_data = create_data()
        forecaster = LSTMForecaster(past_seq_len=24,
                                    input_feature_num=2,
                                    output_feature_num=2,
                                    loss="mae",
                                    lr=0.01)
        forecaster.fit(train_data, epochs=2)
        # quantization with tunning
        forecaster.quantize(train_data, val_data=val_data,
                            metric="mse", relative_drop=0.1, max_trials=3,
                            framework=['onnxrt_qlinearops'])
        pred_q = forecaster.predict_with_onnx(test_data[0], quantize=True)
        eval_q = forecaster.evaluate_with_onnx(test_data, quantize=True)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            ckpt_name_q = os.path.join(tmp_dir_name, "ckpt.q")
            forecaster.export_onnx_file(dirname=ckpt_name, quantized_dirname=ckpt_name_q)

    def test_lstm_forecaster_save_load(self):
        train_data, val_data, test_data = create_data()
        forecaster = LSTMForecaster(past_seq_len=24,
                                    input_feature_num=2,
                                    output_feature_num=2,
                                    loss="mae",
                                    lr=0.01)
        train_mse = forecaster.fit(train_data, epochs=2)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            test_pred_save = forecaster.predict(test_data[0])
            forecaster.save(ckpt_name)
            forecaster.load(ckpt_name)
            test_pred_load = forecaster.predict(test_data[0])
        np.testing.assert_almost_equal(test_pred_save, test_pred_load)

    def test_lstm_forecaster_runtime_error(self):
        train_data, val_data, test_data = create_data()
        forecaster = LSTMForecaster(past_seq_len=24,
                                    input_feature_num=2,
                                    output_feature_num=2,
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

    def test_lstm_forecaster_shape_error(self):
        train_data, val_data, test_data = create_data()
        forecaster = LSTMForecaster(past_seq_len=24,
                                    input_feature_num=2,
                                    output_feature_num=1,
                                    loss="mae",
                                    lr=0.01)
        with pytest.raises(RuntimeError):
            forecaster.fit(train_data, epochs=2)

    def test_lstm_forecaster_xshard_input(self):
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
            forecaster = LSTMForecaster(past_seq_len=24,
                                        input_feature_num=2,
                                        output_feature_num=2,
                                        loss="mae",
                                        lr=0.01,
                                        distributed=distributed)
            forecaster.fit(train_data, epochs=2)
            distributed_pred = forecaster.predict(test_data)
            distributed_eval = forecaster.evaluate(val_data)
        stop_orca_context()

    def test_lstm_forecaster_distributed(self):
        from bigdl.orca import init_orca_context, stop_orca_context
        train_data, val_data, test_data = create_data()
        init_orca_context(cores=4, memory="2g")

        forecaster = LSTMForecaster(past_seq_len=24,
                                    input_feature_num=2,
                                    output_feature_num=2,
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
            local_eval_onnx = forecaster.evaluate_with_onnx(val_data)
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

    def test_lstm_dataloader_distributed(self):
        from bigdl.orca import init_orca_context, stop_orca_context
        train_loader, _, _ = create_data(loader=True)
        init_orca_context(cores=4, memory="2g")
        forecaster = LSTMForecaster(past_seq_len=24,
                                    input_feature_num=2,
                                    output_feature_num=2,
                                    loss="mae",
                                    lr=0.01,
                                    distributed=True)
        forecaster.fit(train_loader, epochs=2)
        stop_orca_context()

    def test_lstm_customized_loss_metric(self):
        from torchmetrics.functional import mean_squared_error
        train_data, _, _ = create_data(loader=True)
        _, _, test_data = create_data()
        loss = torch.nn.L1Loss()
        def customized_metric(y_true, y_pred):
            return mean_squared_error(torch.from_numpy(y_pred),
                                      torch.from_numpy(y_true)).numpy()
        forecaster = LSTMForecaster(past_seq_len=24,
                                    input_feature_num=2,
                                    output_feature_num=2,
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
