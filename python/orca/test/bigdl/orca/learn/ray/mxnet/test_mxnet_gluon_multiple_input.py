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

import numpy as np
import pytest

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from bigdl.orca.learn.mxnet import Estimator, create_config

np.random.seed(1337)  # for reproducibility


def get_train_data_iter(config, kv):
    train_data = [np.random.rand(100, 30), np.random.rand(100, 20)]
    train_label = np.random.randint(0, 10, (200,))
    train = mx.io.NDArrayIter(train_data, train_label,
                              batch_size=config["batch_size"], shuffle=True)
    return train


def get_test_data_iter(config, kv):
    test_data = [np.random.rand(40, 30), np.random.rand(40, 20)]
    test_label = np.random.randint(0, 10, (80,))
    test = mx.io.NDArrayIter(test_data, test_label,
                             batch_size=config["batch_size"], shuffle=True)
    return test


def get_model(config):
    class SimpleModel(gluon.nn.HybridBlock):
        def __init__(self, **kwargs):
            super(SimpleModel, self).__init__(**kwargs)
            self.fc1 = nn.Dense(20)
            self.fc2 = nn.Dense(40)
            self.fc3 = nn.Dense(10)

        def hybrid_forward(self, F, x1, x2):
            y1 = self.fc1(x1)
            y2 = self.fc2(x2)
            y = F.concat(y1, y2, dim=1)
            return self.fc3(y)

    net = SimpleModel()
    net.initialize(mx.init.Xavier(rnd_type="gaussian"), ctx=[mx.cpu()], force_reinit=True)
    return net


def get_loss(config):
    return gluon.loss.SoftmaxCrossEntropyLoss()


def get_metrics(config):
    return ['accuracy', mx.metric.TopKAccuracy(3)]


class TestMXNetGluonMultipleInput(TestCase):
    def test_gluon_multiple_input(self):
        config = create_config(log_interval=2, optimizer="adagrad", seed=1128,
                               optimizer_params={'learning_rate': 0.02})
        estimator = Estimator.from_mxnet(config=config,
                                         model_creator=get_model,
                                         loss_creator=get_loss,
                                         eval_metrics_creator=get_metrics,
                                         validation_metrics_creator=get_metrics,
                                         num_workers=4)
        estimator.fit(get_train_data_iter, validation_data=get_test_data_iter, epochs=2)
        estimator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
