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
from unittest import TestCase

import numpy as np
import pytest

import torch
import torch.nn as nn

from bigdl.orca import OrcaContext
from bigdl.orca.data.pandas import read_csv
from bigdl.orca.learn.metrics import Accuracy

from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.data import SparkXShards
from bigdl.orca.data.image.utils import chunks


np.random.seed(1337)  # for reproducibility
resource_path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "../../../resources")


class LinearDataset(torch.utils.data.Dataset):
    """y = a * x + b"""

    def __init__(self, size=1000):
        X1 = torch.randn(size // 2, 50)
        X2 = torch.randn(size // 2, 50) + 1.5
        self.x = torch.cat([X1, X2], dim=0)
        Y1 = torch.zeros(size // 2, 1)
        Y2 = torch.ones(size // 2, 1)
        self.y = torch.cat([Y1, Y2], dim=0)

    def __getitem__(self, index):
        return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y


class IdentityNet(nn.Module):
    def __init__(self):
        super().__init__()
        # need this line to avoid optimizer raise empty variable list
        self.fc1 = nn.Linear(50, 50)

    def forward(self, input_):
        return input_


class MultiInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.out = nn.Linear(50, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input1, input2):
        x = torch.cat((input1, input2), 1)
        x = self.fc1(x)
        x = self.out(x)
        x = self.out_act(x)
        return x


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input1, input2):
        x = torch.stack((input1, input2), dim=1)
        x = self.fc(x)
        x = self.out_act(x).flatten()
        return x


def train_data_loader(config, batch_size):
    train_dataset = LinearDataset(size=config.get("data_size", 1000))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size
    )
    return train_loader


def val_data_loader(config, batch_size):
    val_dataset = LinearDataset(size=config.get("val_size", 400))
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    return validation_loader


def get_model(config):
    return Net()


def get_optimizer(model, config):
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


def get_estimator(workers_per_node=1, model_fn=get_model):
    estimator = Estimator.from_torch(model=model_fn,
                                     optimizer=get_optimizer,
                                     loss=nn.BCELoss(),
                                     metrics=Accuracy(),
                                     config={"lr": 1e-2},
                                     workers_per_node=workers_per_node,
                                     backend="spark")
    return estimator


class TestPyTorchEstimator(TestCase):
    def test_data_creator(self):
        estimator = get_estimator(workers_per_node=2)
        train_stats = estimator.fit(train_data_loader, epochs=2, batch_size=128)
        print(train_stats)
        val_stats = estimator.evaluate(val_data_loader, batch_size=64)
        print(val_stats)
        assert 0 < val_stats["Accuracy"] < 1
        assert estimator.get_model()

    def test_spark_xshards(self):
        from bigdl.dllib.nncontext import init_nncontext
        from bigdl.orca.data import SparkXShards
        estimator = get_estimator(workers_per_node=1)
        sc = init_nncontext()
        x_rdd = sc.parallelize(np.random.rand(4000, 1, 50).astype(np.float32))
        # torch 1.7.1+ requires target size same as output size, which is (batch, 1)
        y_rdd = sc.parallelize(np.random.randint(0, 2, size=(4000, 1, 1)).astype(np.float32))
        rdd = x_rdd.zip(y_rdd).map(lambda x_y: {'x': x_y[0], 'y': x_y[1]})
        train_rdd, val_rdd = rdd.randomSplit([0.9, 0.1])
        train_xshards = SparkXShards(train_rdd)
        val_xshards = SparkXShards(val_rdd)
        train_stats = estimator.fit(train_xshards, batch_size=256, epochs=2)
        print(train_stats)
        val_stats = estimator.evaluate(val_xshards, batch_size=128)
        print(val_stats)

    def test_dataframe_train_eval(self):

        sc = init_nncontext()
        rdd = sc.range(0, 100)
        df = rdd.map(lambda x: (np.random.randn(50).astype(np.float).tolist(),
                                [int(np.random.randint(0, 2, size=()))])
                     ).toDF(["feature", "label"])

        estimator = get_estimator(workers_per_node=2)
        estimator.fit(df, batch_size=4, epochs=2,
                      feature_cols=["feature"],
                      label_cols=["label"])
        estimator.evaluate(df, batch_size=4,
                           feature_cols=["feature"],
                           label_cols=["label"])

    def test_dataframe_shard_size_train_eval(self):
        from bigdl.orca import OrcaContext
        OrcaContext._shard_size = 30
        sc = init_nncontext()
        rdd = sc.range(0, 100)
        df = rdd.map(lambda x: (np.random.randn(50).astype(np.float).tolist(),
                                [int(np.random.randint(0, 2, size=()))])
                     ).toDF(["feature", "label"])

        estimator = get_estimator(workers_per_node=2)
        estimator.fit(df, batch_size=4, epochs=2,
                      feature_cols=["feature"],
                      label_cols=["label"])
        estimator.evaluate(df, batch_size=4,
                           feature_cols=["feature"],
                           label_cols=["label"])

    def test_partition_num_less_than_workers(self):
        sc = init_nncontext()
        rdd = sc.range(200, numSlices=1)
        df = rdd.map(lambda x: (np.random.randn(50).astype(np.float).tolist(),
                                [int(np.random.randint(0, 2, size=()))])
                     ).toDF(["feature", "label"])

        estimator = get_estimator(workers_per_node=2)
        assert df.rdd.getNumPartitions() < estimator.num_workers

        estimator.fit(df, batch_size=4, epochs=2,
                      feature_cols=["feature"],
                      label_cols=["label"])
        estimator.evaluate(df, batch_size=4,
                           feature_cols=["feature"],
                           label_cols=["label"])
        estimator.predict(df, feature_cols=["feature"]).collect()

    def test_dataframe_predict(self):

        sc = init_nncontext()
        rdd = sc.parallelize(range(20))
        df = rdd.map(lambda x: ([float(x)] * 5,
                                [int(np.random.randint(0, 2, size=()))])
                     ).toDF(["feature", "label"])

        estimator = get_estimator(workers_per_node=2,
                                  model_fn=lambda config: IdentityNet())
        result = estimator.predict(df, batch_size=4,
                                   feature_cols=["feature"])
        expr = "sum(cast(feature <> to_array(prediction) as int)) as error"
        assert result.selectExpr(expr).first()["error"] == 0

    def test_xshards_predict(self):

        sc = init_nncontext()
        rdd = sc.range(0, 110).map(lambda x: np.array([x]*50))
        shards = rdd.mapPartitions(lambda iter: chunks(iter, 5)).map(lambda x: {"x": np.stack(x)})
        shards = SparkXShards(shards)

        estimator = get_estimator(workers_per_node=2,
                                  model_fn=lambda config: IdentityNet())
        result_shards = estimator.predict(shards, batch_size=4)
        result = np.concatenate([shard["prediction"] for shard in result_shards.collect()])
        expected_result = np.concatenate([shard["x"] for shard in result_shards.collect()])

        assert np.array_equal(result, expected_result)

    # currently do not support pandas dataframe
    # def test_pandas_dataframe(self):

    #     OrcaContext.pandas_read_backend = "pandas"
    #     file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
    #     data_shard = read_csv(file_path, usecols=[0, 1, 2], dtype={0: np.float32, 1: np.float32,
    #                                                                2: np.float32})

    #     estimator = get_estimator(model_fn=lambda config: SimpleModel())
    #     estimator.fit(data_shard, batch_size=2, epochs=2,
    #                   feature_cols=["user", "item"],
    #                   label_cols=["label"])

    #     estimator.evaluate(data_shard, batch_size=2, feature_cols=["user", "item"],
    #                        label_cols=["label"])
    #     result = estimator.predict(data_shard, batch_size=2, feature_cols=["user", "item"])
    #     result.collect()

    def test_multiple_inputs_model(self):

        sc = init_nncontext()
        rdd = sc.parallelize(range(100))

        from pyspark.sql import SparkSession
        spark = SparkSession(sc)
        df = rdd.map(lambda x: ([float(x)] * 25, [float(x)] * 25,
                                [int(np.random.randint(0, 2, size=()))])
                     ).toDF(["f1", "f2", "label"])

        estimator = get_estimator(workers_per_node=2,
                                  model_fn=lambda config: MultiInputNet())
        estimator.fit(df, batch_size=4, epochs=2,
                      feature_cols=["f1", "f2"],
                      label_cols=["label"])
        estimator.evaluate(df, batch_size=4,
                           feature_cols=["f1", "f2"],
                           label_cols=["label"])
        result = estimator.predict(df, batch_size=4,
                                   feature_cols=["f1", "f2"])
        result.collect()


if __name__ == "__main__":
    pytest.main([__file__])
