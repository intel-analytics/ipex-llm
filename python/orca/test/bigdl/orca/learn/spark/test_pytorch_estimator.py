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

from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, ArrayType, StructType, StructField

from bigdl.orca import OrcaContext
from bigdl.orca.data.pandas import read_csv
from bigdl.orca.learn.metrics import Accuracy

from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.pytorch.callbacks.base import Callback

import tempfile
import shutil
import logging

np.random.seed(1337)  # for reproducibility
resource_path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "../../resources")


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


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # need this line to avoid optimizer raise empty variable list
        self.fc1 = nn.Linear(1, 1, bias=False)
        self.fc1.weight.data.fill_(1.0)

    def forward(self, input_):
        return self.fc1(input_)


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


class CustomCallback(Callback):

    def on_train_end(self, logs=None):
        assert "train_loss" in logs
        assert "val_loss" in logs
        assert self.model

    def on_epoch_end(self, epoch, logs=None):
        assert "train_loss" in logs
        assert "val_loss" in logs
        assert self.model


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
    torch.manual_seed(0)
    return Net()


def get_optimizer(model, config):
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


def get_estimator(workers_per_node=1, model_fn=get_model, sync_stats=False,
                  log_level=logging.INFO, model_dir=None):
    estimator = Estimator.from_torch(model=model_fn,
                                     optimizer=get_optimizer,
                                     loss=nn.BCELoss(),
                                     metrics=Accuracy(),
                                     config={"lr": 1e-2},
                                     workers_per_node=workers_per_node,
                                     backend="spark",
                                     sync_stats=sync_stats,
                                     model_dir=model_dir,
                                     log_level=log_level)
    return estimator


class TestPyTorchEstimator(TestCase):
    def setUp(self) -> None:
        self.model_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.model_dir)

    def test_spark_xshards_of_dict(self):
        from bigdl.dllib.nncontext import init_nncontext
        from bigdl.orca.data import SparkXShards
        estimator = get_estimator(workers_per_node=1,
                                  model_fn=lambda config: MultiInputNet())
        sc = init_nncontext()
        x1_rdd = sc.parallelize(np.random.rand(4000, 1, 25).astype(np.float32))
        x2_rdd = sc.parallelize(np.random.rand(4000, 1, 25).astype(np.float32))
        # torch 1.7.1+ requires target size same as output size, which is (batch, 1)
        y_rdd = sc.parallelize(np.random.randint(0, 2, size=(4000, 1, 1)).astype(np.float32))
        rdd = x1_rdd.zip(x2_rdd).zip(y_rdd).map(lambda x_y: {'x': [x_y[0][0], x_y[0][1]], 'y': x_y[1]})
        train_rdd, val_rdd = rdd.randomSplit([0.9, 0.1])
        train_xshards = SparkXShards(train_rdd)
        val_xshards = SparkXShards(val_rdd)
        train_stats = estimator.fit(train_xshards, validation_data=val_xshards,
                                    batch_size=256, epochs=2)
        print(train_stats)
        val_stats = estimator.evaluate(val_xshards, batch_size=128)
        print(val_stats)

    def test_pandas_dataframe(self):
        OrcaContext.pandas_read_backend = "pandas"
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        data_shard = read_csv(file_path, usecols=[0, 1, 2], dtype={0: np.float32, 1: np.float32,
                                                                   2: np.float32})
        
        estimator = get_estimator(model_fn=lambda config: SimpleModel())
        estimator.fit(data_shard, batch_size=2, epochs=2,
                      validation_data=data_shard,
                      feature_cols=["user", "item"],
                      label_cols=["label"])

        estimator.evaluate(data_shard, batch_size=2, feature_cols=["user", "item"],
                           label_cols=["label"])
        result = estimator.predict(data_shard, batch_size=2, feature_cols=["user", "item"])
        result.collect()

    def test_dataframe_shard_size_train_eval(self):
        from bigdl.orca import OrcaContext
        OrcaContext._shard_size = 30
        sc = init_nncontext()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100)
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(),
                                  [float(np.random.randint(0, 2, size=()))])
                       )
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])
        df = spark.createDataFrame(data=data, schema=schema)

        estimator = get_estimator(workers_per_node=2)
        train_worker_stats = estimator.fit(df, batch_size=4, epochs=2,
                                           feature_cols=["feature"],
                                           label_cols=["label"])
        # Total samples for one epoch
        assert train_worker_stats[0]["num_samples"] == 100
        eval_worker_stats = estimator.evaluate(df, batch_size=4,
                                               feature_cols=["feature"],
                                               label_cols=["label"],
                                               reduce_results=False, profile=True)
        acc = [stat["Accuracy"].data.item() for stat in eval_worker_stats]
        loss = [stat["val_loss"] for stat in eval_worker_stats]
        validation_time = [stat["profile"]["mean_validation_s"] for stat in eval_worker_stats]
        forward_time = [stat["profile"]["mean_eval_fwd_s"] for stat in eval_worker_stats]
        from bigdl.orca.learn.pytorch.utils import process_stats
        agg_worker_stats = process_stats(eval_worker_stats)
        assert round(agg_worker_stats["Accuracy"].data.item(), 4) == \
               round(sum(acc) / 2, 4)
        assert round(agg_worker_stats["val_loss"], 4) == round(sum(loss) / 2, 4)
        assert round(agg_worker_stats["profile"]["mean_validation_s"], 4) == \
               round(sum(validation_time) / 2, 4)
        assert round(agg_worker_stats["profile"]["mean_eval_fwd_s"], 4) == \
               round(sum(forward_time) / 2, 4)
        assert agg_worker_stats["num_samples"] == 100

        # Test stats given model dir
        estimator2 = get_estimator(workers_per_node=2, model_dir=self.model_dir)
        train_worker_stats = estimator2.fit(df, batch_size=4, epochs=2,
                                            feature_cols=["feature"],
                                            label_cols=["label"])
        assert train_worker_stats[0]["num_samples"] == 100

    def test_tensorboard_callback(self):
        from bigdl.orca.learn.pytorch.callbacks.tensorboard import TensorBoardCallback
        sc = OrcaContext.get_spark_context()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100)
        epochs = 2
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(),
                                  [float(np.random.randint(0, 2, size=()))])
                       )
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])
        df = spark.createDataFrame(data=data, schema=schema)
        df = df.cache()

        estimator = get_estimator(workers_per_node=2, log_level=logging.DEBUG)

        try:
            temp_dir = tempfile.mkdtemp()
            log_dir = os.path.join(temp_dir, "runs_epoch")

            callbacks = [
                TensorBoardCallback(log_dir=log_dir, freq="epoch")
            ]
            estimator.fit(df, batch_size=4, epochs=epochs,
                          callbacks=callbacks,
                          feature_cols=["feature"],
                          label_cols=["label"])

            assert len(os.listdir(log_dir)) > 0

            log_dir = os.path.join(temp_dir, "runs_batch")

            callbacks = [
                TensorBoardCallback(log_dir=log_dir, freq="batch")
            ]
            estimator.fit(df, batch_size=4, epochs=epochs,
                          callbacks=callbacks,
                          feature_cols=["feature"],
                          label_cols=["label"])

            assert len(os.listdir(log_dir)) > 0
        finally:
            shutil.rmtree(temp_dir)

        estimator.shutdown()

    def test_train_max_steps(self):
        from bigdl.orca import OrcaContext
        OrcaContext._shard_size = 30
        sc = init_nncontext()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100)
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(),
                                  [float(np.random.randint(0, 2, size=()))])
                       )
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])
        df = spark.createDataFrame(data=data, schema=schema)

        estimator = get_estimator(workers_per_node=2)
        train_worker_stats = estimator.fit(df, batch_size=4, max_steps=40,
                                           feature_cols=["feature"],
                                           label_cols=["label"])
        # Total samples for last epoch
        assert train_worker_stats[1]["num_samples"] == 60

    
if __name__ == "__main__":
    pytest.main([__file__])
