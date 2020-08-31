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
# Some portions of this file Copyright 2018 Uber Technologies, Inc
# and licensed under the Apache License, Version 2.0
#

# This file is adapted from https://github.com/horovod/horovod/blob/master/examples/pytorch_mnist.py

from __future__ import print_function

import argparse

import horovod.torch as hvd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms

from zoo.ray import RayContext
from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.horovod import HorovodRayRunner


def run_horovod():
    # Temporary patch this script until the MNIST dataset download issue get resolved
    # https://github.com/pytorch/vision/issues/1938
    import urllib
    try:
        # For python 2
        class AppURLopener(urllib.FancyURLopener):
            version = "Mozilla/5.0"

        urllib._urlopener = AppURLopener()
    except AttributeError:
        # For python 3
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    lr = 0.01
    momentum = 0.5
    seed = 43
    log_interval = 10
    fp16_allreduce = False
    use_adasum = False

    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(seed)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {}
    train_dataset = \
        datasets.MNIST('data-%d' % hvd.rank(), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)

    test_dataset = \
        datasets.MNIST('data-%d' % hvd.rank(), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                              sampler=test_sampler, **kwargs)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)

    model = Net()

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not use_adasum else 1

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(model.parameters(), lr=lr * lr_scaler,
                          momentum=momentum)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if use_adasum else hvd.Average)

    def train(epoch):
        model.train()
        # Horovod: set epoch to sampler for shuffling.
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_sampler),
                    100. * batch_idx / len(train_loader), loss.item()))

    def metric_average(val, name):
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def test():
        model.eval()
        test_loss = 0.
        test_accuracy = 0.
        for data, target in test_loader:
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        test_loss /= len(test_sampler)
        test_accuracy /= len(test_sampler)

        # Horovod: average metric values across workers.
        test_loss = metric_average(test_loss, 'avg_loss')
        test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                test_loss, 100. * test_accuracy))

    for epoch in range(1, epochs + 1):
        train(epoch)
        test()


parser = argparse.ArgumentParser()
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The mode for the Spark cluster.')
parser.add_argument("--slave_num", type=int, default=2,
                    help="The number of slave nodes to be used in the cluster."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--cores", type=int, default=8,
                    help="The number of cpu cores you want to use on each node. "
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--memory", type=str, default="10g",
                    help="The size of slave(executor)'s memory you want to use."
                         "You can change it depending on your own cluster setting.")


if __name__ == "__main__":

    args = parser.parse_args()
    num_nodes = 1 if args.cluster_mode == "local" else args.slave_num
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores, num_nodes=num_nodes,
                      memory=args.memory)

    runner = HorovodRayRunner(RayContext.get())
    runner.run(func=run_horovod)
    stop_orca_context()
