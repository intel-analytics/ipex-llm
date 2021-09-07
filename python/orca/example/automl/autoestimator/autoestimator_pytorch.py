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

import numpy as np
import argparse

import torch
import torch.nn as nn
from zoo.orca.automl.auto_estimator import AutoEstimator
from zoo.orca.automl.pytorch_utils import LR_NAME
from zoo.orca import init_orca_context, stop_orca_context


class Net(nn.Module):
    def __init__(self, dropout, fc1_size, fc2_size):
        super().__init__()
        self.fc1 = nn.Linear(50, fc1_size)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(fc2_size, 1)
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


def model_creator(config):
    return Net(dropout=config["dropout"],
               fc1_size=config["fc1_size"],
               fc2_size=config["fc2_size"])


def get_optimizer(model, config):
    return torch.optim.SGD(model.parameters(), lr=config["lr"])


def get_train_val_data():
    def get_x_y(size):
        input_size = 50
        x1 = np.random.randn(size // 2, input_size)
        x2 = np.random.randn(size // 2, input_size) + 1.5
        x = np.concatenate([x1, x2], axis=0)
        y1 = np.zeros((size // 2, 1))
        y2 = np.ones((size // 2, 1))
        y = np.concatenate([y1, y2], axis=0)
        return x, y

    train_data = get_x_y(size=1000)
    val_data = get_x_y(size=400)
    return train_data, val_data


def create_linear_search_space():
    from zoo.orca.automl import hp
    return {
        "dropout": hp.uniform(0.2, 0.3),
        "fc1_size": hp.choice([50, 64]),
        "fc2_size": hp.choice([100, 128]),
        LR_NAME: hp.choice([0.001, 0.003, 0.01]),
        "batch_size": hp.choice([32, 64])
    }


def train_example(args):
    auto_est = AutoEstimator.from_torch(
        model_creator=model_creator,
        optimizer="Adam",
        loss="BCELoss",
        logs_dir="/tmp/zoo_automl_logs",
        resources_per_trial={"cpu": args.cpus_per_trial},
        name="test_fit")
    train_data, val_data = get_train_val_data()
    auto_est.fit(data=train_data,
                 epochs=args.epochs,
                 validation_data=val_data,
                 metric="accuracy",
                 n_sampling=args.trials,
                 search_space=create_linear_search_space())
    # Choose the best model
    best_model = auto_est.get_best_model()

    y_hat = best_model(torch.from_numpy(val_data[0]).float()).detach().numpy()
    from zoo.orca.automl.metrics import Evaluator
    accuracy = Evaluator.evaluate(metric="accuracy", y_true=val_data[1], y_pred=y_hat)
    print("Evaluate: accuracy is", accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Autoestimator_pytorch',
        description='Automatically fit the model and return the best model.')
    parser.add_argument('--cluster_mode',
                        type=str,
                        default="local",
                        help='The mode for the Spark cluster.')
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="The number of nodes to be used in the cluster. "
        "You can change it depending on your own cluster setting.")
    parser.add_argument(
        "--cores",
        type=int,
        default=4,
        help="The number of cpu cores you want to use on each node. "
        "You can change it depending on your own cluster setting.")
    parser.add_argument(
        "--memory",
        type=str,
        default="10g",
        help="The memory you want to use on each node. "
        "You can change it depending on your own cluster setting.")
    parser.add_argument("--workers_per_node",
                        type=int,
                        default=2,
                        help="The number of workers to run on each node")
    parser.add_argument('--k8s_master',
                        type=str,
                        default="",
                        help="The k8s master. "
                        "It should be k8s://https://<k8s-apiserver-host>: "
                        "<k8s-apiserver-port>.")
    parser.add_argument("--container_image",
                        type=str,
                        default="",
                        help="The runtime k8s image. "
                        "You can change it with your k8s image.")
    parser.add_argument('--k8s_driver_host',
                        type=str,
                        default="",
                        help="The k8s driver localhost. ")
    parser.add_argument('--k8s_driver_port',
                        type=str,
                        default="",
                        help="The k8s driver port.")
    parser.add_argument(
        '--cpus_per_trial',
        type=int,
        default=2,
        help="The number of cores you want to allocate for each trial.")
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help="The number of epochs in each trial.")
    parser.add_argument('--trials',
                        type=int,
                        default=4,
                        help="The number of searching trials.")

    args = parser.parse_args()
    if args.cluster_mode == "local":
        init_orca_context(cluster_mode="local",
                          cores=args.cores,
                          num_nodes=args.num_nodes,
                          memory=args.memory,
                          init_ray_on_spark=True)
    elif args.cluster_mode == "yarn":
        init_orca_context(cluster_mode="yarn-client",
                          cores=args.cores,
                          memory=args.memory,
                          init_ray_on_spark=True)
    elif args.cluster_mode == "k8s":
        if not args.k8s_master or not args.container_image \
                or not args.k8s_driver_host or not args.k8s_driver_port:
            parser.print_help()
            parser.error('k8s_master, container_image,'
                         'k8s_driver_host/port are required not to be empty')
        init_orca_context(cluster_mode="k8s",
                          master=args.k8s_master,
                          container_image=args.container_image,
                          cores=args.cores,
                          init_ray_on_spark=True,
                          conf={
                              "spark.driver.host": args.k8s_driver_host,
                              "spark.driver.port": args.k8s_driver_port
                          })

    train_example(args)
    stop_orca_context()
