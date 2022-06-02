import os
import argparse

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy

from model import model_creator, optimizer_creator

parser = argparse.ArgumentParser(description='PyTorch Tensorboard Example')
parser.add_argument('--cluster_mode', type=str, default="yarn-client",
                    help='The cluster mode, such as local, yarn-client, yarn-cluster, spark-submit or k8s.')
parser.add_argument('--backend', type=str, default="spark",
                    help='The backend of PyTorch Estimator; '
                         'bigdl, ray and spark are supported.')
parser.add_argument('--batch_size', type=int, default=4, help='The training batch size')
parser.add_argument('--epochs', type=int, default=2, help='The number of epochs to train for')
parser.add_argument('--data_dir', type=str, default="./data", help='The path of dataset')
parser.add_argument('--remote_dir', type=str, help='The path to load data from remote resources like HDFS or S3')
parser.add_argument('--download', default=True, action='store_true', help='Download dataset or not')
parser.add_argument('--no-download', dest='download', action='store_false', help='Download dataset or not')
parser.add_argument('--extra-python-lib', type=str, default='model.py', help='Load dependency when running on yarn')
args = parser.parse_args()

if args.cluster_mode == "yarn-cluster":
    from bigdl.orca.data.file import get_remote_file_to_local
    get_remote_file_to_local(args.remote_dir, args.data_dir)

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.FashionMNIST(root=args.data_dir, train=True,
                                             download=args.download, transform=transform)

testset = torchvision.datasets.FashionMNIST(root=args.data_dir, train=False,
                                            download=args.download, transform=transform)

def train_data_creator(config, batch_size):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    return trainloader

def validation_data_creator(config, batch_size):
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    return testloader

def main():
    init_orca_context(cluster_mode=args.cluster_mode, extra_python_lib=args.extra_python_lib)

    criterion = nn.CrossEntropyLoss()
    orca_estimator = Estimator.from_torch(model=model_creator,
                                          optimizer=optimizer_creator,
                                          loss=criterion,
                                          metrics=[Accuracy()],# Orca validation methods for evaluate.
                                          model_dir=os.getcwd(),# The path to save model.
                                          backend=args.backend)

    train_stats = orca_estimator.fit(train_data_creator, epochs=args.epochs, batch_size=args.batch_size)
    print("Train stats: {}".format(train_stats))

    eval_stats = orca_estimator.evaluate(validation_data_creator, batch_size=args.batch_size)
    print("Train stats: {}".format(eval_stats))

    stop_orca_context()

if __name__ == '__main__':
    main()