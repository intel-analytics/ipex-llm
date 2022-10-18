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
# ==============================================================================
# Most of the pytorch code is adapted from Pytorch's tutorial for
# neural networks training on Cifar10
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#

from __future__ import print_function
import os
from os.path import exists
from os import makedirs
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca.learn.trigger import EveryEpoch


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The cluster mode, such as local, yarn, spark-submit or k8s.')
parser.add_argument('--runtime', type=str, default="spark",
                    help='The runtime backend, one of spark or ray.')
parser.add_argument('--address', type=str, default="",
                    help='The cluster address if the driver connects to an existing ray cluster. '
                         'If it is empty, a new Ray cluster will be created.')
parser.add_argument('--backend', type=str, default="bigdl",
                    help='The backend of PyTorch Estimator; '
                         'bigdl, ray and spark are supported')
parser.add_argument('--batch_size', type=int, default=4, help='The training batch size')
parser.add_argument('--epochs', type=int, default=2, help='The number of epochs to train for')
parser.add_argument('--data_dir', type=str, default="./data", help='The path to dataset')
parser.add_argument('--download', type=bool, default=True, help='Download dataset or not')
parser.add_argument("--executor_memory", type=str, default="5g", help="executor memory")
parser.add_argument("--driver_memory", type=str, default="5g", help="driver memory")
parser.add_argument("--wandb_callback", type=bool, default=False,
                    help='Whether to enable WandbLoggerCallback. Only for ray and spark backend')
args = parser.parse_args()

if args.runtime == "ray":
    init_orca_context(runtime=args.runtime, address=args.address)
else:
    if args.cluster_mode == "local":
        init_orca_context(memory="4g")
    elif args.cluster_mode.startswith("yarn"):
        if args.cluster_mode == "yarn-client":
            init_orca_context(cluster_mode="yarn-client")
        elif args.cluster_mode == "yarn-cluster":
            init_orca_context(cluster_mode="yarn-cluster",
                              memory=args.executor_memory, driver_memory=args.driver_memory)
    elif args.cluster_mode == "spark-submit":
        init_orca_context(cluster_mode="spark-submit")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def train_loader_creator(config, batch_size):
    trainset = torchvision.datasets.CIFAR10(root=config.get("root", "./data"), train=True,
                                            download=args.download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    return trainloader


def test_loader_creator(config, batch_size):
    testset = torchvision.datasets.CIFAR10(root=config.get("root", "./data"), train=False,
                                           download=args.download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    return testloader


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def model_creator(config):
    net = Net()
    return net


def optim_creator(model, config):
    optimizer = optim.SGD(model.parameters(),
                          lr=config.get("lr", 0.001),
                          momentum=config.get("momentum", 0.9))
    return optimizer


criterion = nn.CrossEntropyLoss()
batch_size = args.batch_size
root_dir = args.data_dir
if not exists(root_dir):
    makedirs(root_dir)

train_loader = train_loader_creator(config={"root": root_dir}, batch_size=batch_size)
test_loader = test_loader_creator(config={"root": root_dir}, batch_size=batch_size)

# plot some random images
dataiter = iter(train_loader)
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images), one_channel=False)
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

dataiter = iter(test_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images), one_channel=False)
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

if args.backend == "bigdl":
    net = model_creator(config={})
    optimizer = optim_creator(model=net, config={"lr": 0.001})
    orca_estimator = Estimator.from_torch(model=net,
                                          optimizer=optimizer,
                                          loss=criterion,
                                          metrics=[Accuracy()],
                                          backend="bigdl")

    orca_estimator.fit(data=train_loader, epochs=args.epochs, validation_data=test_loader,
                       checkpoint_trigger=EveryEpoch())

    res = orca_estimator.evaluate(data=test_loader)
    print("Accuracy of the network on the test images: %s" % res)
elif args.backend in ["ray", "spark"]:
    orca_estimator = Estimator.from_torch(model=model_creator,
                                          optimizer=optim_creator,
                                          loss=criterion,
                                          metrics=[Accuracy()],
                                          model_dir=os.getcwd(),
                                          backend=args.backend,
                                          use_tqdm=True,
                                          config={"lr": 0.001,
                                                  "root": root_dir})

    fit_args = dict(data=train_loader_creator, epochs=args.epochs, batch_size=batch_size)
    if args.wandb_callback:
        from bigdl.orca.learn.pytorch.callbacks.wandb import WandbLoggerCallback
        wandb_callback = WandbLoggerCallback(project="cifar")
        fit_args.update(dict(callbacks=[wandb_callback]))

    orca_estimator.fit(**fit_args)

    res = orca_estimator.evaluate(data=test_loader_creator)
    for r in res:
        print(r, ":", res[r])
else:
    invalidInputError(False, "Only bigdl, ray, and spark are supported as the backend,"
                      " but got {}".format(args.backend))

stop_orca_context()
