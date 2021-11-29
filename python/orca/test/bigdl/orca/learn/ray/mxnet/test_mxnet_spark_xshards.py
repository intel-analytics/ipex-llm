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

import os.path
import pytest

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from bigdl.orca import OrcaContext
import bigdl.orca.data.pandas
from bigdl.orca.learn.mxnet import Estimator, create_config


def prepare_data_symbol(df):
    data = {'input': np.array(df['data'].values.tolist())}
    label = {'label': df['label'].values}
    return {'x': data, 'y': label}


def prepare_data_gluon(df):
    data = np.array(df['data'].values.tolist())
    label = df['label'].values
    return {'x': data, 'y': label}


def get_loss(config):
    return gluon.loss.SoftmaxCrossEntropyLoss()


def get_gluon_metrics(config):
    return mx.metric.Accuracy()


def get_metrics(config):
    return 'accuracy'


def get_symbol_model(config):
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


def get_gluon_model(config):
    class SimpleModel(gluon.Block):
        def __init__(self, **kwargs):
            super(SimpleModel, self).__init__(**kwargs)
            self.fc1 = nn.Dense(20)
            self.fc2 = nn.Dense(10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    net = SimpleModel()
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=[mx.cpu()])
    return net


class TestMXNetSparkXShards(TestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
        OrcaContext.pandas_read_backend = "pandas"

    def tearDown(self):
        OrcaContext.pandas_read_backend = "spark"

    def test_xshards_symbol_with_val(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")
        train_file_path = os.path.join(resource_path, "orca/learn/single_input_json/train")
        train_data_shard = bigdl.orca.data.pandas.read_json(
            train_file_path, orient='records', lines=False).transform_shard(prepare_data_symbol)
        test_file_path = os.path.join(resource_path, "orca/learn/single_input_json/test")
        test_data_shard = bigdl.orca.data.pandas.read_json(
            test_file_path, orient='records', lines=False).transform_shard(prepare_data_symbol)
        config = create_config(log_interval=1, seed=42)
        estimator = Estimator.from_mxnet(config=config, model_creator=get_symbol_model,
                                         validation_metrics_creator=get_metrics,
                                         eval_metrics_creator=get_metrics, num_workers=2)
        estimator.fit(train_data_shard, epochs=2)
        train_data_shard2 = bigdl.orca.data.pandas.read_json(
            train_file_path, orient='records', lines=False).transform_shard(prepare_data_symbol)
        estimator.fit(train_data_shard2, validation_data=test_data_shard, epochs=1, batch_size=32)
        estimator.shutdown()

    def test_xshards_symbol_without_val(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")
        train_file_path = os.path.join(resource_path, "orca/learn/single_input_json/train")
        train_data_shard = bigdl.orca.data.pandas.read_json(
            train_file_path, orient='records', lines=False).transform_shard(prepare_data_symbol)
        config = create_config(log_interval=1, seed=42)
        estimator = Estimator.from_mxnet(config=config, model_creator=get_symbol_model,
                                         eval_metrics_creator=get_metrics, num_workers=2)
        estimator.fit(train_data_shard, epochs=2, batch_size=16)
        estimator.shutdown()

    def test_xshards_gluon(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")
        train_file_path = os.path.join(resource_path, "orca/learn/single_input_json/train")
        train_data_shard = bigdl.orca.data.pandas.read_json(
            train_file_path, orient='records', lines=False).transform_shard(prepare_data_gluon)
        test_file_path = os.path.join(resource_path, "orca/learn/single_input_json/train")
        test_data_shard = bigdl.orca.data.pandas.read_json(
            test_file_path, orient='records', lines=False).transform_shard(prepare_data_gluon)
        config = create_config(log_interval=1, seed=42)
        estimator = Estimator.from_mxnet(config=config, model_creator=get_gluon_model,
                                         loss_creator=get_loss,
                                         validation_metrics_creator=get_gluon_metrics,
                                         eval_metrics_creator=get_gluon_metrics,
                                         num_workers=2)
        estimator.fit(train_data_shard, validation_data=test_data_shard, epochs=2, batch_size=8)
        estimator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
