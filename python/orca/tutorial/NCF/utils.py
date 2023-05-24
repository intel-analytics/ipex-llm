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

import os
import argparse
import json
import tempfile

from bigdl.dllib.utils.file_utils import is_local_path
from bigdl.orca.data.file import put_local_file_to_remote, get_remote_file_to_local
from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext


def parse_args(description, mode="train"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_dir", type=str, default="./",
                        help="The path to load data from local or remote resources.")
    parser.add_argument("--dataset", type=str, default="ml-1m",
                        choices=("ml-1m", "ml-100k"),
                        help="The name of the dataset. ml-1m or ml-100k")
    parser.add_argument("--model_dir", type=str, default="./",
                        help="The path to save model and logs.")
    parser.add_argument("--cluster_mode", type=str, default="local",
                        choices=("local", "yarn-client", "yarn-cluster", "k8s-client",
                                 "k8s-cluster", "spark-submit", "bigdl-submit"),
                        help="The cluster mode, one of local, yarn-client, yarn-cluster, "
                             "k8s-client, k8s-cluster, spark-submit or bigdl-submit.")
    parser.add_argument("--backend", type=str, default="spark",
                        choices=("spark", "ray"),
                        help="The backend of Orca Estimator, either ray or spark.")
    parser.add_argument("--workers_per_node", type=int, default=1,
                        help="The number of workers on each node.")
    if mode == "train":
        parser.add_argument("--tensorboard", action='store_true',
                            help="Whether to use TensorBoard as the train callback.")

        parser.add_argument("--lr_scheduler", action='store_true',
                            help="Whether to use learning rate scheduler for training.")
    args = parser.parse_args()
    return args


def init_orca(cluster_mode, extra_python_lib=None):
    if cluster_mode == "local":
        sc = init_orca_context(cluster_mode="local")
    elif cluster_mode.startswith("yarn"):
        if cluster_mode == "yarn-client":
            sc = init_orca_context(cluster_mode="yarn-client",
                                   cores=4, memory="10g", num_nodes=2,
                                   extra_python_lib=extra_python_lib)
        elif cluster_mode == "yarn-cluster":
            sc = init_orca_context(cluster_mode="yarn-cluster",
                                   cores=4, memory="10g", num_nodes=2,
                                   extra_python_lib=extra_python_lib)
    elif cluster_mode.startswith("k8s"):
        if cluster_mode == "k8s-client":
            conf = {
                "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
                ".options.claimName": "nfsvolumeclaim",
                "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
                ".mount.path": "/bigdl/nfsdata"
            }
            sc = init_orca_context(cluster_mode="k8s-client",
                                   cores=4, memory="10g", num_nodes=2,
                                   extra_python_lib=extra_python_lib,
                                   master=os.environ.get("RUNTIME_SPARK_MASTER"),
                                   container_image=os.environ.get("RUNTIME_K8S_SPARK_IMAGE"),
                                   conf=conf)
        elif cluster_mode == "k8s-cluster":
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
            sc = init_orca_context(cluster_mode="k8s-cluster",
                                   cores=4, memory="10g", num_nodes=2,
                                   extra_python_lib=extra_python_lib,
                                   master=os.environ.get("RUNTIME_SPARK_MASTER"),
                                   container_image=os.environ.get("RUNTIME_K8S_SPARK_IMAGE"),
                                   penv_archive="file:///bigdl/nfsdata/environment.tar.gz",
                                   conf=conf)
    elif cluster_mode == "bigdl-submit":
        sc = init_orca_context(cluster_mode="bigdl-submit")
    elif cluster_mode == "spark-submit":
        sc = init_orca_context(cluster_mode="spark-submit")
    else:
        exit("cluster_mode should be one of 'local', 'yarn-client', "
             "'yarn-cluster', 'k8s-client', 'k8s-cluster', 'bigdl-submit' or 'spark-submit', "
             "but got " + cluster_mode)
    return sc


def schedule_func(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * 0.5


def get_pytorch_callbacks(args):
    if args.tensorboard:
        from bigdl.orca.learn.pytorch.callbacks.tensorboard import TensorBoardCallback
        return [TensorBoardCallback(log_dir=os.path.join(args.model_dir, "logs"),
                                    freq=1000)]
    else:
        return []


def save_model_config(config, model_dir, file_name="config.json"):
    if is_local_path(model_dir):  # save to local path
        with open(os.path.join(model_dir, file_name), "w") as f:
            json.dump(config, f)
    else:  # save to remote path
        with tempfile.TemporaryDirectory() as tmpdirname:
            local_path = os.path.join(tmpdirname, file_name)
            remote_path = os.path.join(model_dir, file_name)
            with open(local_path, "w") as f:
                json.dump(config, f)
            put_local_file_to_remote(local_path, remote_path)


def load_model_config(model_dir, file_name="config.json"):
    if is_local_path(model_dir):  # load from local path
        with open(os.path.join(model_dir, file_name), "r") as f:
            config = json.load(f)
    else:  # load from remote path
        with tempfile.TemporaryDirectory() as tmpdirname:
            local_path = os.path.join(tmpdirname, file_name)
            remote_path = os.path.join(model_dir, file_name)
            get_remote_file_to_local(remote_path=remote_path, local_path=local_path)
            with open(local_path, "r") as f:
                config = json.load(f)
    return config
