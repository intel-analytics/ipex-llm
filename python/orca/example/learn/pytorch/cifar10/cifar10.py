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
# ==============================================================================
# Most of the pytorch code is adapted from Pytorch's tutorial for
# neural networks training on Cifar10
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#

from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.pytorch import Estimator
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.trigger import EveryEpoch


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The cluster mode, such as local, yarn or k8s.')
parser.add_argument('--backend', type=str, default="bigdl",
                    help='The backend of PyTorch Estimator; '
                         'bigdl and torch_distributed are supported')
args = parser.parse_args()

if args.cluster_mode == "local":
    init_orca_context(memory="4g")
elif args.cluster_mode == "yarn":
    init_orca_context(
        cluster_mode="yarn-client", num_nodes=2, driver_memory="4g",
        conf={"spark.rpc.message.maxSize": "1024",
              "spark.task.maxFailures": "1",
              "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def train_loader_creator(config, batch_size):
    trainset = torchvision.datasets.CIFAR10(root=config.get("root", "./data"), train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    return trainloader


def test_loader_creator(config, batch_size):
    testset = torchvision.datasets.CIFAR10(root=config.get("root", "./data"), train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return testloader


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


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
batch_size = 4
root_dir = "./data"

train_loader = train_loader_creator(config={"root": root_dir}, batch_size=batch_size)
test_loader = test_loader_creator(config={"root": root_dir}, batch_size=batch_size)

# plot some random images
dataiter = iter(train_loader)
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

dataiter = iter(test_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

if args.backend == "bigdl":
    net = model_creator(config={})
    optimizer = optim_creator(model=net, config={"lr": 0.001})
    orca_estimator = Estimator.from_torch(model=net,
                                          optimizer=optimizer,
                                          loss=criterion,
                                          metrics=[Accuracy()],
                                          backend="bigdl")

    orca_estimator.fit(data=train_loader, epochs=2, validation_data=test_loader,
                       checkpoint_trigger=EveryEpoch())

    res = orca_estimator.evaluate(data=test_loader)
    print("Accuracy of the network on the test images: %s" % res)
elif args.backend == "torch_distributed":
    orca_estimator = Estimator.from_torch(model=model_creator,
                                          optimizer=optim_creator,
                                          loss=criterion,
                                          metrics=[Accuracy()],
                                          backend="torch_distributed",
                                          config={"lr": 0.001,
                                                  "root": root_dir})

    orca_estimator.fit(data=train_loader_creator, epochs=2, batch_size=batch_size)

    res = orca_estimator.evaluate(data=test_loader_creator)
    for r in res:
        print(r, ":", res[r])
else:
    raise NotImplementedError("Only bigdl and torch_distributed are supported as the backend,"
                              " but got {}".format(args.backend))

stop_orca_context()
