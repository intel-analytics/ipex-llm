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
import argparse

import torch
import torch.nn as nn

from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator


class LinearDataset(torch.utils.data.Dataset):
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

def get_model(config):
    torch.manual_seed(0)
    return Net()

def get_optimizer(model, config):
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))

parser = argparse.ArgumentParser()
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The cluster mode, such as local, ray,  yarn, spark-submit or k8s.')

if __name__ == "__main__":

    args = parser.parse_args()
    if args.cluster_mode == "ray":
        init_orca_context("ray", address="172.168.0.204:6379")
    elif args.cluster_mode == "local":
        init_orca_context()
    elif args.cluster_mode.startswith("yarn"):
        init_orca_context(cluster_mode=args.cluster_mode, cores=4, num_nodes=2)
    elif args.cluster_mode == "spark-submit":
        init_orca_context(cluster_mode=args.cluster_mode)

    """
    Orca Pytorch Estimator
    """
    estimator = Estimator.from_torch(model=get_model,
                                    optimizer=get_optimizer,
                                    loss=nn.BCELoss(),
                                    metrics=Accuracy(),
                                    config={"lr": 1e-2},
                                    workers_per_node=2,
                                    backend="torch_distributed",
                                    sync_stats=False)

    train_results = estimator.fit(data=train_data_loader,
        epochs=1,
        batch_size=32)
    print("This is Train Results:", train_results)

    val_results = estimator.evaluate(data=val_data_loader,
        batch_size=32)
    print("This is Val Results:", val_results)

    stop_orca_context()