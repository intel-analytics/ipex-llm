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
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from unittest import TestCase
import pytest
import tempfile

from bigdl.chronos.autots.model.auto_nbeats import AutoNBeats
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.automl import hp

input_feature_dim = 1
output_feature_dim = 1
past_seq_len = 24
future_seq_len = 5


def get_x_y(size):
    x = np.random.randn(size, past_seq_len, input_feature_dim)
    y = np.random.randn(size, future_seq_len, output_feature_dim)
    return x.astype(np.float32), y.astype(np.float32)


class RandomDataset(Dataset):
    def __init__(self, size=1000):
        x, y = get_x_y(size)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_dataloader_creator(config):
    return DataLoader(RandomDataset(size=1000),
                      batch_size=config["batch_size"],
                      shuffle=True)


def valid_dataloader_creator(config):
    return DataLoader(RandomDataset(size=400),
                      batch_size=config["batch_size"],
                      shuffle=True)


def get_auto_estimator():
    auto_nbeats = AutoNBeats(past_seq_len=past_seq_len,
                             future_seq_len=future_seq_len,
                             stack_types=('generic', 'generic'),
                             nb_blocks_per_stack=3,
                             thetas_dim=(4, 8),
                             share_weights_in_stack=True,
                             hidden_layer_units=hp.grid_search([64, 128]),
                             nb_harmonics=None,
                             loss=torch.nn.MSELoss(),
                             metric="mse",
                             optimizer='Adam',
                             lr=hp.choice([0.001, 0.003, 0.01]),
                             dropout=hp.uniform(0.1, 0.2),
                             logs_dir="/tmp/auto_lstm",
                             cpus_per_trial=2,
                             name="auto_lstm")
    return auto_nbeats


class TestAutoNBEATS(TestCase):
    def setUp(self) -> None:
        init_orca_context(cores=4, memory="4g", init_ray_on_spark=True)

    def tearDown(self) -> None:
        stop_orca_context()
    
    def test_fit_np(self):
        auto_nbeats_ = get_auto_estimator()
        auto_nbeats_.fit(data=(train_dataloader_creator(config={'batch_size': 64})),
                         epochs=1,
                         batch_size=hp.choice([32, 64]),
                         validation_data=valid_dataloader_creator(config={'batch_size': 64}),
                         n_sampling=1)
        assert auto_nbeats_.get_best_model()
        best_config = auto_nbeats_.get_best_config()
        assert 0.1 <= best_config['dropout'] <= 0.2
        assert best_config['batch_size'] in (32, 64)
    
    def test_fit_data_creator(self):
        auto_nbeats_ = get_auto_estimator()
        auto_nbeats_.fit(data=train_dataloader_creator,
                         epochs=1,
                         batch_size=hp.choice([32, 64]),
                         validation_data=valid_dataloader_creator,
                         n_sampling=1)
        assert auto_nbeats_.get_best_model()
        best_config = auto_nbeats_.get_best_config()
        assert 0.1 <= best_config['dropout'] <= 0.2
        assert best_config['batch_size'] in (32, 64)
        assert 64 <= best_config['hidden_layer_units'] <= 128
    
    def test_predict_evalution(self):
        auto_nbeats_ = get_auto_estimator()
        auto_nbeats_.fit(data=train_dataloader_creator(config={"batch_size": 64}),
                         epochs=1,
                         validation_data=valid_dataloader_creator(config={"batch_size": 64}),
                         n_sampling=1)
        test_data_x, test_data_y = get_x_y(size=100)
        auto_nbeats_.predict(test_data_x)
        auto_nbeats_.evaluate((test_data_x, test_data_y))

    def test_onnx_methods(self):
        auto_nbeats_ = get_auto_estimator()
        auto_nbeats_.fit(data=train_dataloader_creator(config={"batch_size": 64}),
                      epochs=1,
                      validation_data=valid_dataloader_creator(config={"batch_size": 64}),
                      n_sampling=1)
        test_data_x, test_data_y = get_x_y(size=100)
        pred = auto_nbeats_.predict(test_data_x)
        eval_res = auto_nbeats_.evaluate((test_data_x, test_data_y))
        try:
            import onnx
            import onnxruntime
            pred_onnx = auto_nbeats_.predict_with_onnx(test_data_x)
            eval_res_onnx = auto_nbeats_.evaluate_with_onnx((test_data_x, test_data_y))
            np.testing.assert_almost_equal(pred, pred_onnx, decimal=5)
            np.testing.assert_almost_equal(eval_res, eval_res_onnx, decimal=5)
        except ImportError:
            pass

    def test_save_load(self):
        auto_nbeats_ = get_auto_estimator()
        auto_nbeats_.fit(data=train_dataloader_creator(config={'batch_size': 64}),
                         epochs=1,
                         validation_data=valid_dataloader_creator(config={'batch_size': 64}),
                         n_sampling=1)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            auto_nbeats_.save(tmp_dir_name)
            auto_nbeats_.load(tmp_dir_name)
        test_data_x, test_data_y = get_x_y(size=100)
        pred = auto_nbeats_.predict(test_data_x)
        eval_res = auto_nbeats_.evaluate((test_data_x, test_data_y))
        try:
            import onnx
            import onnxruntime
            pred_onnx = auto_nbeats_.predict_with_onnx(test_data_x)
            eval_res_onnx = auto_nbeats_.evaluate_with_onnx((test_data_x, test_data_y))
            np.testing.assert_almost_equal(pred, pred_onnx, decimal=5)
            np.testing.assert_almost_equal(eval_res, eval_res_onnx, decimal=5)
        except ImportError:
            pass

if __name__ == "__main__":
    pytest.main([__file__])
