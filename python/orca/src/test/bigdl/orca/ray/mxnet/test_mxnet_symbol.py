#
# Copyright 2018 Analytics Zoo Authors.
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

import numpy as np
import pytest

import mxnet as mx
from zoo.ray.mxnet import MXNetTrainer, create_trainer_config

np.random.seed(1337)  # for reproducibility


def get_data_iters(config, kv):
    train_data = np.random.rand(200, 30)
    train_label = np.random.randint(0, 10, (200,))
    train = mx.io.NDArrayIter({"input": train_data}, {"label": train_label},
                              batch_size=config["batch_size"], shuffle=True)
    test_data = np.random.rand(80, 30)
    test_label = np.random.randint(0, 10, (80,))
    test = mx.io.NDArrayIter({"input": test_data}, {"label": test_label},
                             batch_size=config["batch_size"], shuffle=True)
    return train, test


def get_model(config):
    input_data = mx.symbol.Variable('input')
    y_true = mx.symbol.Variable('label')
    fc1 = mx.symbol.FullyConnected(data=input_data, num_hidden=20, name='fc1')
    fc2 = mx.symbol.FullyConnected(data=fc1, num_hidden=10, name='fc2')
    output = mx.symbol.SoftmaxOutput(data=fc2, label=y_true, name='output')
    mod = mx.mod.Module(symbol=output,
                        data_names=['input'],
                        label_names=['label'],
                        context=mx.cpu())
    return mod


def get_metrics(config):
    return 'accuracy'


class TestMXNetSymbol(TestCase):
    def test_symbol(self):
        config = create_trainer_config(batch_size=32, log_interval=2, seed=42)
        trainer = MXNetTrainer(config, get_data_iters, get_model, metrics_creator=get_metrics)
        trainer.train(nb_epoch=2)


if __name__ == "__main__":
    pytest.main([__file__])
