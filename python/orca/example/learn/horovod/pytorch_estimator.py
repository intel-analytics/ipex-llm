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

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.pytorch import Estimator


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


def train_data_creator(config, batch_size):
    train_dataset = LinearDataset(2, 5, size=config.get("data_size", 1000))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size
    )
    return train_loader


def validation_data_creator(config, batch_size):
    val_dataset = LinearDataset(2, 5, size=config.get("val_size", 400))
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    return validation_loader


def train_example(workers_per_node):
    estimator = Estimator.from_torch(
        model=model_creator,
        optimizer=optimizer_creator,
        loss=nn.MSELoss(),
        scheduler_creator=scheduler_creator,
        workers_per_node=workers_per_node,
        config={
            "lr": 1e-2,  # used in optimizer_creator
            "hidden_size": 1  # used in model_creator
        }, backend="horovod")

    # train 5 epochs
    stats = estimator.fit(train_data_creator, batch_size=4, epochs=5)
    print("train stats: {}".format(stats))
    val_stats = estimator.evaluate(validation_data_creator)
    print("validation stats: {}".format(val_stats))

    # retrieve the model
    model = estimator.get_model()
    print("trained weight: % .2f, bias: % .2f" % (
        model.weight.item(), model.bias.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The mode for the Spark cluster.')
    parser.add_argument("--num_nodes", type=int, default=1,
                        help="The number of nodes to be used in the cluster. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--cores", type=int, default=4,
                        help="The number of cpu cores you want to use on each node. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--memory", type=str, default="10g",
                        help="The memory you want to use on each node. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--workers_per_node", type=int, default=2,
                        help="The number of workers to run on each node")
    parser.add_argument('--k8s_master', type=str, default="",
                        help="The k8s master. "
                             "It should be k8s://https://<k8s-apiserver-host>: "
                             "<k8s-apiserver-port>.")
    parser.add_argument("--container_image", type=str, default="",
                        help="The runtime k8s image. "
                             "You can change it with your k8s image.")
    parser.add_argument('--k8s_driver_host', type=str, default="",
                        help="The k8s driver localhost.")
    parser.add_argument('--k8s_driver_port', type=str, default="",
                        help="The k8s driver port.")

    args = parser.parse_args()
    if args.cluster_mode == "local":
        init_orca_context(cluster_mode="local", cores=args.cores,
                          num_nodes=args.num_nodes, memory=args.memory)
    elif args.cluster_mode == "yarn":
        init_orca_context(cluster_mode="yarn-client", cores=args.cores,
                          num_nodes=args.num_nodes, memory=args.memory)
    elif args.cluster_mode == "k8s":
        if not args.k8s_master or not args.container_image \
                or not args.k8s_driver_host or not args.k8s_driver_port:
            parser.print_help()
            parser.error('k8s_master, container_image,'
                         'k8s_driver_host/port are required not to be empty')
        init_orca_context(cluster_mode="k8s", master=args.k8s_master,
                          container_image=args.container_image,
                          num_nodes=args.num_nodes, cores=args.cores,
                          conf={"spark.driver.host": args.k8s_driver_host,
                                "spark.driver.port": args.k8s_driver_port})
    train_example(workers_per_node=args.workers_per_node)
    stop_orca_context()
