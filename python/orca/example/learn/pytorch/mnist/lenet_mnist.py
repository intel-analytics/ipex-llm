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

from __future__ import print_function
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.pytorch import Estimator
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.trigger import EveryEpoch


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dir', default='/tmp/data', metavar='N',
                        help='the folder store mnist data')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training per executor(default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing per executor(default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The mode for the Spark cluster. local or yarn.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dir, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False)

    if args.cluster_mode == "local":
        init_orca_context(cores=1, memory="20g")
    elif args.cluster_mode == "yarn":
        init_orca_context(
            cluster_mode="yarn-client", cores=4, num_nodes=2, memory="2g",
            driver_memory="10g", driver_cores=1,
            conf={"spark.rpc.message.maxSize": "1024",
                  "spark.task.maxFailures": "1",
                  "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})

    model = LeNet()
    model.train()
    criterion = nn.NLLLoss()

    adam = torch.optim.Adam(model.parameters(), args.lr)
    zoo_estimator = Estimator.from_torch(model=model, optimizer=adam, loss=criterion,
                                         backend="bigdl")
    zoo_estimator.fit(data=train_loader, epochs=args.epochs, validation_data=test_loader,
                      validation_methods=[Accuracy()], checkpoint_trigger=EveryEpoch())
    zoo_estimator.evaluate(data=test_loader, validation_methods=[Accuracy()])
    stop_orca_context()


if __name__ == '__main__':
    main()
