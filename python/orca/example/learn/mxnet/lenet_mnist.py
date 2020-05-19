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

# Reference: https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/image/mnist.html

import argparse

from zoo import init_spark_on_local, init_spark_on_yarn
from zoo.ray import RayContext
from zoo.orca.learn.mxnet import MXNetTrainer, create_trainer_config


def get_data_iters(config, kv):
    import os
    import zipfile
    import mxnet as mx
    from bigdl.dataset.base import maybe_download

    # In order to avoid conflict where multiple workers on the same node download and
    # zip data under the same location, here we let each worker have its own folder.

    # Not using mxnet.test_utils.get_mnist_iterator directly because data path is
    # hard-coded in this function.

    # In practice, data is supposed to be stored on a file system accessible to workers on
    # all nodes, for example, on HDFS or S3.
    maybe_download("mnist.zip", "worker" + str(kv.rank),
                   "http://data.mxnet.io/mxnet/data/mnist.zip")
    if not os.path.isdir("worker" + str(kv.rank) + "/data"):
        with zipfile.ZipFile("worker" + str(kv.rank) + "/mnist.zip") as zf:
            zf.extractall("worker" + str(kv.rank) + "/data")

    train_iter = mx.io.MNISTIter(
        image="worker" + str(kv.rank) + "/data/train-images-idx3-ubyte",
        label="worker" + str(kv.rank) + "/data/train-labels-idx1-ubyte",
        input_shape=(1, 28, 28),
        batch_size=config["batch_size"],
        shuffle=True,
        flat=False,
        num_parts=kv.num_workers,
        part_index=kv.rank)
    val_iter = mx.io.MNISTIter(
        image="worker" + str(kv.rank) + "/data/t10k-images-idx3-ubyte",
        label="worker" + str(kv.rank) + "/data/t10k-labels-idx1-ubyte",
        input_shape=(1, 28, 28),
        batch_size=config["batch_size"],
        flat=False,
        num_parts=kv.num_workers,
        part_index=kv.rank)
    return train_iter, val_iter


def get_model(config):
    import mxnet as mx
    from mxnet import gluon
    from mxnet.gluon import nn
    import mxnet.ndarray as F

    class LeNet(gluon.Block):
        def __init__(self, **kwargs):
            super(LeNet, self).__init__(**kwargs)
            with self.name_scope():
                # layers created in name_scope will inherit name space
                # from parent layer.
                self.conv1 = nn.Conv2D(20, kernel_size=(5, 5))
                self.pool1 = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
                self.conv2 = nn.Conv2D(50, kernel_size=(5, 5))
                self.pool2 = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
                self.fc1 = nn.Dense(500)
                self.fc2 = nn.Dense(10)

        def forward(self, x):
            x = self.pool1(F.tanh(self.conv1(x)))
            x = self.pool2(F.tanh(self.conv2(x)))
            # 0 means copy over size from corresponding dimension.
            # -1 means infer size from the rest of dimensions.
            x = x.reshape((0, -1))
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
            return x

    net = LeNet()
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=[mx.cpu()])
    return net


def get_loss(config):
    from mxnet import gluon
    return gluon.loss.SoftmaxCrossEntropyLoss()


def get_metrics(config):
    import mxnet as mx
    return mx.metric.Accuracy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a LeNet model for handwritten digit recognition.')
    parser.add_argument('--hadoop_conf', type=str,
                        help='The path to the hadoop configuration folder. Required if you '
                             'wish to run on yarn clusters. Otherwise, run in local mode.')
    parser.add_argument('--conda_name', type=str,
                        help='The name of conda environment. Required if you '
                             'wish to run on yarn clusters.')
    parser.add_argument('--executor_cores', type=int, default=4,
                        help='The number of executor cores you want to use.')
    parser.add_argument('-n', '--num_workers', type=int, default=2,
                        help='The number of MXNet workers to be launched.')
    parser.add_argument('-s', '--num_servers', type=int,
                        help='The number of MXNet servers to be launched. If not specified, '
                        'default to be equal to the number of workers.')
    parser.add_argument('-b', '--batch_size', type=int, default=100,
                        help='The number of samples per gradient update for each worker.')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='The number of epochs to train the model.')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.02,
                        help='Learning rate for the LeNet model.')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='The number of batches to wait before logging throughput and '
                             'metrics information during the training process.')
    opt = parser.parse_args()

    if opt.hadoop_conf:
        assert opt.conda_name is not None, "conda_name must be specified for yarn mode"
        sc = init_spark_on_yarn(
            hadoop_conf=opt.hadoop_conf,
            conda_name=opt.conda_name,
            num_executor=opt.num_workers,
            executor_cores=opt.executor_cores)
    else:
        sc = init_spark_on_local(cores="*")
    ray_ctx = RayContext(sc=sc)
    ray_ctx.init()

    config = create_trainer_config(opt.batch_size, optimizer="sgd",
                                   optimizer_params={'learning_rate': opt.learning_rate},
                                   log_interval=opt.log_interval, seed=42)
    trainer = MXNetTrainer(config, data_creator=get_data_iters, model_creator=get_model,
                           loss_creator=get_loss, metrics_creator=get_metrics,
                           num_workers=opt.num_workers, num_servers=opt.num_servers)
    trainer.train(nb_epoch=opt.epochs)
    ray_ctx.stop()
    sc.stop()
