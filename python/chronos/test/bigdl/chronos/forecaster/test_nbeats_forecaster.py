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

import tempfile
import torch
import numpy as np
from unittest import TestCase
import pytest

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.chronos.forecaster.nbeats_forecaster import NBeatsForecaster
from bigdl.chronos.metric.forecast_metrics import Evaluator


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


class TestChronosNbeatsForecaster(TestCase):
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
                                      metircs=['mae'],
                                      lr=0.01)
        forecaster.fit((train_data[0], train_data[1]), epochs=2)
        # inference
        nbeats_pred=forecaster.predict(test_data[0])
        assert nbeats_pred.shape == test_data[1].shape
        eva = forecaster.evaluate(test_data)
        assert eva[0].shape == test_data[1].shape[1:]

    def test_nbeats_forecaster_data_loader(self):
        train_loader, _, _ = create_data(loader=True)
        init_orca_context(cores=4, memory="4g", init_ray_on_spark=True)
        forecater = NBeatsForecaster(past_seq_len=24,
                                     future_seq_len=5,
                                     distributed=True)
        forecater.fit(train_loader, epochs=2)
        stop_orca_context()


