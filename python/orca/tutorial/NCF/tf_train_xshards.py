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
import math
import tensorflow as tf

from tf_model import ncf_model
from process_xshards import prepare_data

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2 import Estimator

parser = argparse.ArgumentParser(description="Tensorflow NCF Training with Orca Xshards")
parser.add_argument("--data_dir", type=str,
                    help="The path to load data from local or remote resources.")
parser.add_argument("--cluster_mode", type=str, default="local",
                    help="The cluster mode, such as local, yarn-client, yarn-cluster, "
                         "k8s-client, k8s-cluster, spark-submit or bigdl-submit.")
parser.add_argument("--backend", type=str, default="spark", help="ray or spark")
parser.add_argument("--tensorboard", action='store_true',
                    help="Whether to use TensorBoardCallback.")
args = parser.parse_args()


# Step 1: Init Orca Context
if args.cluster_mode == "local":
    sc = init_orca_context(cluster_mode="local", cores=4)
elif args.cluster_mode.startswith("yarn"):
    if args.cluster_mode == "yarn-client":
        sc = init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2,
                               driver_cores=2, driver_memory="2g",
                               extra_python_lib="tf_model.py")
    elif args.cluster_mode == "yarn-cluster":
        sc = init_orca_context(cluster_mode="yarn-cluster", cores=4, memory="10g", num_nodes=2,
                               driver_cores=2, driver_memory="2g",
                               extra_python_lib="tf_model.py")
elif args.cluster_mode.startswith("k8s"):
    if args.cluster_mode == "k8s-client":
        conf = {
            "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".options.claimName": "nfsvolumeclaim",
            "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
            ".mount.path": "/bigdl/nfsdata"
        }
        sc = init_orca_context(cluster_mode="k8s-client", num_nodes=2, cores=4, memory="10g",
                               driver_cores=2, driver_memory="2g",
                               master=os.environ.get("RUNTIME_SPARK_MASTER"),
                               container_image=os.environ.get("RUNTIME_K8S_SPARK_IMAGE"),
                               extra_python_lib="tf_model.py",
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
        sc = init_orca_context(cluster_mode="k8s-cluster", num_nodes=2, cores=4, memory="10g",
                               driver_cores=2, driver_memory="2g",
                               master=os.environ.get("RUNTIME_SPARK_MASTER"),
                               container_image=os.environ.get("RUNTIME_K8S_SPARK_IMAGE"),
                               penv_archive="file:///bigdl/nfsdata/environment.tar.gz",
                               extra_python_lib="tf_model.py",
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


# Step 2: Read and process data using Orca Xshards
train_data, test_data, user_num, item_num, sparse_feats_input_dims, num_dense_feats, \
    feature_cols, label_cols = prepare_data(args.data_dir, num_ng=4)


# Step 3: Define the NCF model
config = dict(
    factor_num=16,
    lr=1e-2,
    item_num=item_num,
    user_num=user_num,
    dropout=0.5,
    sparse_feats_input_dims=sparse_feats_input_dims,
    num_dense_feats=num_dense_feats,
    sparse_feats_embed_dims=8,
    num_layers=3
)


def model_creator(config):
    model = ncf_model(user_num=config["user_num"],
                      item_num=config["item_num"],
                      num_layers=config["num_layers"],
                      factor_num=config["factor_num"],
                      dropout=config["dropout"],
                      lr=config["lr"],
                      sparse_feats_input_dims=config["sparse_feats_input_dims"],
                      sparse_feats_embed_dims=config["sparse_feats_embed_dims"],
                      num_dense_feats=config["num_dense_feats"])
    return model


# Step 4: Distributed training with Orca TF2 Estimator
est = Estimator.from_keras(model_creator=model_creator,
                           config=config,
                           backend=args.backend)

batch_size = 10240
train_steps = math.ceil(len(train_data) / batch_size)
val_steps = math.ceil(len(test_data) / batch_size)
callbacks = [tf.keras.callbacks.TensorBoard(log_dir="./log")] if args.tensorboard else []

est.fit(train_data,
        epochs=2,
        batch_size=batch_size,
        feature_cols=feature_cols,
        label_cols=label_cols,
        steps_per_epoch=train_steps,
        callbacks=callbacks)


# Step 5: Distributed evaluation of the trained model
result = est.evaluate(test_data,
                      feature_cols=feature_cols,
                      label_cols=label_cols,
                      batch_size=batch_size,
                      num_steps=val_steps)
print("Evaluation results:")
for r in result:
    print("{}: {}".format(r, result[r]))


# Step 6: Save the trained TensorFlow model and processed data for resuming training or prediction
est.save("NCF_model")
train_data.save_pickle(os.path.join(args.data_dir, "train_xshards"))
test_data.save_pickle(os.path.join(args.data_dir, "test_xshards"))


# Step 7: Stop Orca Context when program finishes
stop_orca_context()
