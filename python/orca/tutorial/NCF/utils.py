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

from bigdl.orca import init_orca_context


def parse_args(framework, input_data_type):
    parser = argparse.ArgumentParser(description="{} NCF Training with {}".format(framework,
                                                                                  input_data_type))
    parser.add_argument("--data_dir", type=str,
                        help="The path to load data from local or remote resources.")
    parser.add_argument("--cluster_mode", type=str, default="local",
                        help="The cluster mode, such as local, yarn-client, yarn-cluster, "
                             "k8s-client, k8s-cluster, spark-submit or bigdl-submit.")
    parser.add_argument("--backend", type=str, default="spark", help="ray or spark")
    parser.add_argument("--tensorboard", action='store_true',
                        help="Whether to use TensorBoardCallback.")

    if framework == "PyTorch":
        parser.add_argument("--workers_per_node", type=int, default=1,
                            help="The number of PyTorch workers on each node.")

    args = parser.parse_args()
    return args


def init_orca(args, framework, input_data_type):
    if args.cluster_mode.startswith("yarn") or args.cluster_mode.startswith("k8s"):
        cores = 4
        memory = "8g"
        num_nodes = 2
        driver_cores = 2
        driver_memory = "2g"

        if framework == "PyTorch":
            extra_python_lib = "pytorch_model.py"
        else:
            extra_python_lib = "tf_model.py"
        if input_data_type == "DataLoader":
            extra_python_lib += ",pytorch_dataset.py"
        elif input_data_type == "Orca Xshards":
            extra_python_lib += ",process_xshards.py"

    if args.cluster_mode == "local":
        sc = init_orca_context(cluster_mode="local")
    elif args.cluster_mode.startswith("yarn"):
        if args.cluster_mode == "yarn-client":
            sc = init_orca_context(cluster_mode="yarn-client",
                                   cores=cores, memory=memory, num_nodes=num_nodes,
                                   driver_cores=driver_cores, driver_memory=driver_memory,
                                   extra_python_lib=extra_python_lib)
        elif args.cluster_mode == "yarn-cluster":
            sc = init_orca_context(cluster_mode="yarn-cluster",
                                   cores=cores, memory=memory, num_nodes=num_nodes,
                                   driver_cores=driver_cores, driver_memory=driver_memory,
                                   extra_python_lib=extra_python_lib)
    elif args.cluster_mode.startswith("k8s"):
        if args.cluster_mode == "k8s-client":
            conf = {
                "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
                ".options.claimName": "nfsvolumeclaim",
                "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim"
                ".mount.path": "/bigdl/nfsdata"
            }
            sc = init_orca_context(cluster_mode="k8s-client",
                                   cores=cores, memory=memory, num_nodes=num_nodes,
                                   driver_cores=driver_cores, driver_memory=driver_memory,
                                   extra_python_lib=extra_python_lib,
                                   master=os.environ.get("RUNTIME_SPARK_MASTER"),
                                   container_image=os.environ.get("RUNTIME_K8S_SPARK_IMAGE"),
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
            sc = init_orca_context(cluster_mode="k8s-cluster",
                                   cores=cores, memory=memory, num_nodes=num_nodes,
                                   driver_cores=driver_cores, driver_memory=driver_memory,
                                   extra_python_lib=extra_python_lib,
                                   master=os.environ.get("RUNTIME_SPARK_MASTER"),
                                   container_image=os.environ.get("RUNTIME_K8S_SPARK_IMAGE"),
                                   penv_archive="file:///bigdl/nfsdata/environment.tar.gz",
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
