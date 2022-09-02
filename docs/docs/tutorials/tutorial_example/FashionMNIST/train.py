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
# Most of the Pytorch code is adapted from Pytorch's tutorial for
# visualizing training with tensorboard
# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
#

import os
import argparse

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca.data.file import get_remote_file_to_local

from model import model_creator, optimizer_creator

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--cluster_mode', type=str, default="spark-submit",
                    help='The cluster mode, such as local, yarn-client, yarn-cluster, '
                         'spark-submit, bigdl-submit or k8s.')
parser.add_argument('--remote_dir', type=str, help='The path to load data from remote resources')
args = parser.parse_args()

def train_data_creator(config, batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    if args.remote_dir is not None:
        data_dir = "/tmp/dataset"
        get_remote_file_to_local(remote_path=args.remote_dir, local_path=data_dir)
        trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True,
                                                     download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
    else:
        print("Please specify the dataset path on a network file system.")
    return trainloader

def main():
    if args.cluster_mode.startswith("yarn"):
        if args.cluster_mode == "yarn-client":
            init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2,
                              driver_cores=2, driver_memory="4g",
                              extra_python_lib="model.py")
        elif args.cluster_mode == "yarn-cluster":
            init_orca_context(cluster_mode="yarn-cluster", cores=4, memory="10g", num_nodes=2,
                              driver_cores=2, driver_memory="4g",
                              extra_python_lib="model.py")
    elif args.cluster_mode == "bigdl-submit":
        init_orca_context(cluster_mode="bigdl-submit")
    elif args.cluster_mode == "spark-submit":
        init_orca_context(cluster_mode="spark-submit")
    else:
        print("init_orca_context failed. cluster_mode should be one of 'yarn', 'bigdl-submit' or 'spark-submit' " 
              "but got " + args.cluster_mode)

    criterion = nn.CrossEntropyLoss()
    model_dir = args.remote_dir
    orca_estimator = Estimator.from_torch(model=model_creator,
                                          optimizer=optimizer_creator,
                                          loss=criterion,
                                          metrics=[Accuracy()],
                                          model_dir=model_dir,
                                          backend="spark")

    train_stats = orca_estimator.fit(train_data_creator, epochs=1, batch_size=32)
    print("Train stats: {}".format(train_stats))

    stop_orca_context()

if __name__ == '__main__':
    main()
