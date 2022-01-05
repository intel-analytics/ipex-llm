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

from unittest import TestCase
from bigdl.chronos.model.nbeats_pytorch import model_creator
from bigdl.nano.pytorch.trainer import Trainer
import numpy as np
import tempfile
import os
import torch


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


class TestNbeatsPytorch(TestCase):
    train_data, val_data, test_data = create_data(loader=True)

    def test_fit(self):
        model = model_creator({"past_seq_len": 24,
                               "future_seq_len": 5})
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(model,
                                   loss=torch.nn.MSELoss(),
                                   optimizer=torch.optim.Adam(model.parameters(), lr=0.005))
        trainer.fit(pl_model, self.train_data)
