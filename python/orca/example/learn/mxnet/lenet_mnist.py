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

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.mxnet import Estimator, create_config


def get_train_data_iter(config, kv):
    from mxnet.test_utils import get_mnist_iterator
    from filelock import FileLock
    with FileLock("data.lock"):
        iters = get_mnist_iterator(config["batch_size"], (1, 28, 28),
                                   num_parts=kv.num_workers, part_index=kv.rank)
        return iters[0]


def get_test_data_iter(config, kv):
    from mxnet.test_utils import get_mnist_iterator
    from filelock import FileLock
    with FileLock("data.lock"):
        iters = get_mnist_iterator(config["batch_size"], (1, 28, 28),
                                   num_parts=kv.num_workers, part_index=kv.rank)
        return iters[1]


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
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The mode for the Spark cluster.')
    parser.add_argument('--cores', type=int, default=4,
                        help='The number of cores you want to use on each node.')
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
    parser.add_argument('--log_interval', type=int, default=20,
                        help='The number of batches to wait before logging throughput and '
                             'metrics information during the training process.')
    opt = parser.parse_args()

    num_nodes = 1 if opt.cluster_mode == "local" else opt.num_workers
    init_orca_context(cluster_mode=opt.cluster_mode, cores=opt.cores, num_nodes=num_nodes)

    config = create_config(optimizer="sgd",
                           optimizer_params={'learning_rate': opt.learning_rate},
                           log_interval=opt.log_interval, seed=42)
    estimator = Estimator.from_mxnet(config=config, model_creator=get_model,
                                     loss_creator=get_loss, validation_metrics_creator=get_metrics,
                                     num_workers=opt.num_workers, num_servers=opt.num_servers,
                                     eval_metrics_creator=get_metrics)
    estimator.fit(data=get_train_data_iter, validation_data=get_test_data_iter,
                  epochs=opt.epochs, batch_size=opt.batch_size)
    estimator.shutdown()
    stop_orca_context()
