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

import argparse

import numpy as np
import torch
import torch.nn as nn

from zoo import init_spark_on_yarn, init_spark_on_local
from zoo.ray import RayContext
from zoo.orca.learn.pytorch.pytorch_horovod_estimator import PyTorchHorovodEstimator


class LinearDataset(torch.utils.data.Dataset):
    """y = a * x + b"""

    def __init__(self, a, b, size=1000):
        x = np.arange(0, 10, 10 / size, dtype=np.float32)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(a * x + b)

    def __getitem__(self, index):
        return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)


def model_creator(config):
    """Returns a torch.nn.Module object."""
    return nn.Linear(1, config.get("hidden_size", 1))


def optimizer_creator(model, config):
    """Returns optimizer defined upon the model parameters."""
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


def scheduler_creator(optimizer, config):
    """Returns a learning rate scheduler wrapping the optimizer.
    You will need to set ``TorchTrainer(scheduler_step_freq="epoch")``
    for the scheduler to be incremented correctly.
    If using a scheduler for validation loss, be sure to call
    ``trainer.update_scheduler(validation_loss)``.
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


def data_creator(config):
    """Returns training dataloader, validation dataloader."""
    train_dataset = LinearDataset(2, 5, size=config.get("data_size", 1000))
    val_dataset = LinearDataset(2, 5, size=config.get("val_size", 400))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 32))
    return train_loader, validation_loader


def train_example():
    trainer1 = PyTorchHorovodEstimator(
        model_creator=model_creator,
        data_creator=data_creator,
        optimizer_creator=optimizer_creator,
        loss_creator=nn.MSELoss,
        scheduler_creator=scheduler_creator,
        config={
            "lr": 1e-2,  # used in optimizer_creator
            "hidden_size": 1,  # used in model_creator
            "batch_size": 4,  # used in data_creator
        })

    # train 5 epochs
    for i in range(5):
        stats = trainer1.train()
        print(stats)


parser = argparse.ArgumentParser()
parser.add_argument("--hadoop_conf", type=str,
                    help="turn on yarn mode by passing the path to the hadoop"
                         " configuration folder. Otherwise, turn on local mode.")
parser.add_argument("--slave_num", type=int, default=2,
                    help="The number of slave nodes")
parser.add_argument("--conda_name", type=str,
                    help="The name of conda environment.")
parser.add_argument("--executor_cores", type=int, default=8,
                    help="The number of driver's cpu cores you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--executor_memory", type=str, default="10g",
                    help="The size of slave(executor)'s memory you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--driver_memory", type=str, default="2g",
                    help="The size of driver's memory you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--driver_cores", type=int, default=8,
                    help="The number of driver's cpu cores you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--extra_executor_memory_for_ray", type=str, default="20g",
                    help="The extra executor memory to store some data."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--object_store_memory", type=str, default="4g",
                    help="The memory to store data on local."
                         "You can change it depending on your own cluster setting.")

if __name__ == "__main__":

    args = parser.parse_args()
    if args.hadoop_conf:
        sc = init_spark_on_yarn(
            hadoop_conf=args.hadoop_conf,
            conda_name=args.conda_name,
            num_executor=args.slave_num,
            executor_cores=args.executor_cores,
            executor_memory=args.executor_memory,
            driver_memory=args.driver_memory,
            driver_cores=args.driver_cores,
            extra_executor_memory_for_ray=args.extra_executor_memory_for_ray)
        ray_ctx = RayContext(
            sc=sc,
            object_store_memory=args.object_store_memory)
        ray_ctx.init()
    else:
        sc = init_spark_on_local()
        ray_ctx = RayContext(
            sc=sc,
            object_store_memory=args.object_store_memory)
        ray_ctx.init()
    train_example()
