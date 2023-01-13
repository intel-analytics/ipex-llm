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

# Step 0: Import necessary libraries
import os
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from pytorch_dataset import load_dataset, process_users_items, get_input_dims
from pytorch_model import NCF

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.pytorch.callbacks.tensorboard import TensorBoardCallback
from bigdl.orca.learn.metrics import Accuracy, Precision, Recall

parser = argparse.ArgumentParser(description="PyTorch NCF Training with DataLoader")
parser.add_argument("--data_dir", type=str,
                    help="The path to load data from local or remote resources.")
parser.add_argument("--cluster_mode", type=str, default="local",
                    help="The cluster mode, such as local, yarn-client, yarn-cluster, "
                         "k8s-client, k8s-cluster, spark-submit or bigdl-submit.")
parser.add_argument("--backend", type=str, default="spark", help="ray or spark")
parser.add_argument("--tensorboard", action='store_true',
                    help="Whether to use TensorBoardCallback.")
parser.add_argument("--workers_per_node", type=int, default=1,
                    help="The number of PyTorch workers on each node.")
args = parser.parse_args()


# Step 1: Init Orca Context
if args.cluster_mode == "local":
    sc = init_orca_context(cluster_mode="local")
elif args.cluster_mode.startswith("yarn"):
    if args.cluster_mode == "yarn-client":
        sc = init_orca_context(cluster_mode="yarn-client", cores=4, memory="4g", num_nodes=2,
                               driver_cores=2, driver_memory="2g",
                               extra_python_lib="pytorch_model.py,pytorch_dataset.py")
    elif args.cluster_mode == "yarn-cluster":
        sc = init_orca_context(cluster_mode="yarn-cluster", cores=4, memory="4g", num_nodes=2,
                               driver_cores=2, driver_memory="2g",
                               extra_python_lib="pytorch_model.py,pytorch_dataset.py")
elif args.cluster_mode.startswith("k8s"):
    if args.cluster_mode == "k8s-client":
        conf = {
            "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".options.claimName": "nfsvolumeclaim",
            "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".mount.path": "/bigdl/nfsdata"
        }
        sc = init_orca_context(cluster_mode="k8s-client", num_nodes=2, cores=4, memory="4g",
                               driver_cores=2, driver_memory="2g",
                               master=os.environ.get("RUNTIME_SPARK_MASTER"),
                               container_image=os.environ.get("RUNTIME_K8S_SPARK_IMAGE"),
                               extra_python_lib="pytorch_model.py,pytorch_dataset.py",
                               conf=conf)
    elif args.cluster_mode == "k8s-cluster":
        conf = {
            "spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".options.claimName": "nfsvolumeclaim",
            "spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".mount.path": "/bigdl/nfsdata",
            "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".options.claimName": "nfsvolumeclaim",
            "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".mount.path": "/bigdl/nfsdata",
            "spark.kubernetes.authenticate.driver.serviceAccountName": "spark",
            "spark.kubernetes.file.upload.path": "/bigdl/nfsdata/"
        }
        sc = init_orca_context(cluster_mode="k8s-cluster", num_nodes=2, cores=4, memory="4g",
                               driver_cores=2, driver_memory="2g",
                               master=os.environ.get("RUNTIME_SPARK_MASTER"),
                               container_image=os.environ.get("RUNTIME_K8S_SPARK_IMAGE"),
                               penv_archive="file:///bigdl/nfsdata/environment.tar.gz",
                               extra_python_lib="pytorch_model.py,pytorch_dataset.py",
                               conf=conf)
elif args.cluster_mode == "bigdl-submit":
    sc = init_orca_context(cluster_mode="bigdl-submit")
elif args.cluster_mode == "spark-submit":
    sc = init_orca_context(cluster_mode="spark-submit")
else:
    print("cluster_mode should be one of 'local', 'yarn-client', "
          "'yarn-cluster', 'k8s-client', 'k8s-cluster', 'bigdl-submit' or 'spark-submit', "
          "but got " + args.cluster_mode)
    exit()


# Step 2: Define train and test datasets as PyTorch DataLoader
def train_loader_func(config, batch_size):
    train_dataset, _ = load_dataset(config["dataset_dir"], config["num_ng"])
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=0)
    return train_loader


def test_loader_func(config, batch_size):
    _, test_dataset = load_dataset(config["dataset_dir"], config["num_ng"])
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=0)
    return test_loader


# Step 3: Define the model, optimizer and loss
def model_creator(config):
    users, items, user_num, item_num, sparse_features, dense_features, \
        total_cols = process_users_items(config["dataset_dir"])
    sparse_feats_input_dims, num_dense_feats = get_input_dims(users, items,
                                                              sparse_features, dense_features)
    model = NCF(user_num=user_num,
                item_num=item_num,
                factor_num=config["factor_num"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                model=config["model"],
                sparse_feats_input_dims=sparse_feats_input_dims,
                sparse_feats_embed_dims=config["sparse_feats_embed_dims"],
                num_dense_feats=num_dense_feats)
    model.train()
    return model


def optimizer_creator(model, config):
    return optim.Adam(model.parameters(), lr=config["lr"])

loss = nn.BCEWithLogitsLoss()


# Step 4: Distributed training with Orca PyTorch Estimator
callbacks = [TensorBoardCallback(log_dir="runs", freq=1000)] if args.tensorboard else []

est = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator,
                           loss=loss,
                           metrics=[Accuracy(), Precision(), Recall()],
                           backend=args.backend,
                           use_tqdm=True,
                           workers_per_node=args.workers_per_node,
                           config={"dataset_dir": args.data_dir,
                                   "num_ng": 4,
                                   "factor_num": 16,
                                   "num_layers": 3,
                                   "dropout": 0.5,
                                   "lr": 0.01,
                                   "model": "NeuMF-end",
                                   "sparse_feats_embed_dims": 8})
est.fit(data=train_loader_func, epochs=2, batch_size=10240, callbacks=callbacks)


# Step 5: Distributed evaluation of the trained model
result = est.evaluate(data=test_loader_func, batch_size=10240)
print("Evaluation results:")
for r in result:
    print("{}: {}".format(r, result[r]))


# Step 6: Save the trained PyTorch model
est.save("NCF_model")


# Step 7: Stop Orca Context when program finishes
stop_orca_context()
