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
from unittest import TestCase

import ray
from ray.data import Dataset

import torch
import torch.nn as nn
import torch.optim as optim

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy

def train_data_creator(a=5, b=10, size=1000):
    def get_dataset(a, b, size) -> Dataset:
        items = [i / size for i in range(size)]
        dataset = ray.data.from_items([{
            "x": x,
            "y": a * x + b
        } for x in items])
        return dataset

    ray_dataset = get_dataset(a, b, size)
    return ray_dataset

def model_creator(config):
    net = nn.Linear(1, 1)
    net = net.double()
    return net

def optim_creator(model, config):
    optimizer = optim.SGD(model.parameters(),
                          lr=config.get("lr", 0.001),
                          momentum=config.get("momentum", 0.9))
    return optimizer

class TestPytorchEstimator(TestCase):
    def setUp(self):
        init_orca_context(runtime="ray", address="localhost:6379")

    def tearDown(self):
        stop_orca_context()

    def test_train(self):
        dataset = train_data_creator()
        orca_estimator = Estimator.from_torch(model=model_creator,
                                            optimizer=optim_creator,
                                            loss=nn.MSELoss(),
                                            metrics=[Accuracy()],
                                            config={"lr": 0.001},
                                            workers_per_node=2,
                                            backend="torch_distributed",
                                            sync_stats=True)
        train_stats = orca_estimator.fit(data=dataset, 
                                         epochs=2, 
                                         batch_size=32, 
                                         label_cols="y")

        assert orca_estimator.get_model()

if __name__ == "__main__":
    pytest.main([__file__])
