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

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--cluster_mode', type=str, default="spark-submit",
                    help='The cluster mode, such as local, yarn-client, yarn-cluster, spark-submit or k8s.')
parser.add_argument('--remote_dir', type=str, help='The path to load data from remote resources like HDFS or S3')
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
                              extra_python_lib="./orca_example.zip")
        elif args.cluster_mode == "yarn-cluster":
            init_orca_context(cluster_mode="yarn-cluster", cores=4, memory="10g", num_nodes=2,
                              driver_cores=2, driver_memory="4g",
                              extra_python_lib="./orca_example.zip")
    elif args.cluster_mode == "bigdl-submit":
        init_orca_context(cluster_mode="bigdl-submit")
    elif args.cluster_mode == "spark-submit":
        init_orca_context(cluster_mode="spark-submit")
    else:
        print("init_orca_context failed. cluster_mode should be one of 'yarn', 'bigdl-submit' or 'spark-submit' " 
              "but got " + args.cluster_mode)

    from FashionMNIST.model import model_creator, optimizer_creator

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