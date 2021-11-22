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
from __future__ import print_function
import argparse
import torch
import torchvision
import os
from torchvision import datasets, transforms
from bigdl.dllib.optim.optimizer import *
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.torch import TorchModel, TorchLoss
from bigdl.dllib.estimator import *
from bigdl.dllib.nncontext import *
from bigdl.dllib.feature.common import FeatureSet
from bigdl.dllib.keras.metrics import Accuracy, Top5Accuracy
from bigdl.dllib.utils.utils import detect_conda_env_name

import math

model_names = sorted(name for name in torchvision.models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(torchvision.models.__dict__[name]))


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--max_epochs', default=90, type=int, metavar='N',
                        help='number of max epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cores', default=4, type=int,
                        help='num of CPUs to use.')
    parser.add_argument('--nodes', default=1, type=int,
                        help='num of nodes to use.')
    parser.add_argument('--executor_memory', default='20g', type=str,
                        help='size of executor memory.')
    parser.add_argument('--driver_memory', default='20g', type=str,
                        help='size of driver memory.')
    parser.add_argument('--driver_cores', default=1, type=int,
                        help='num of driver cores to use.')
    parser.add_argument("--num_executors", type=int, default=16,\
                        help="number of executors")
    parser.add_argument("--deploy_mode", type=str, default="yarn-client",
                        help="yarn deploy mode, yarn-client or yarn-cluster")
    args = parser.parse_args()

    # init
    hadoop_conf = os.environ.get("HADOOP_CONF_DIR")
    assert hadoop_conf, "Directory path to hadoop conf not found for yarn-client mode. Please " \
            "set the environment variable HADOOP_CONF_DIR"

    sc = init_orca_context(cluster_mode=args.deploy_mode, hadoop_conf=hadoop_conf,
        conf={"spark.executor.memory": args.executor_memory,
                "spark.executor.cores": args.cores,
                "spark.executor.instances": args.num_executors
    })

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    list = os.listdir(args.data)
    print(list)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    model = torchvision.models.resnet50()
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False)

    iterationPerEpoch = int(math.ceil(float(152) / args.batch_size))
    step = Step(iterationPerEpoch * 30, 0.1)
    zooOptimizer = SGD(args.lr, momentum=args.momentum, dampening=0.0,
                       leaningrate_schedule=step, weightdecay=args.weight_decay)
    zooModel = TorchModel.from_pytorch(model)
    criterion = torch.nn.CrossEntropyLoss()
    zooCriterion = TorchLoss.from_pytorch(criterion)
    estimator = Estimator(zooModel, optim_methods=zooOptimizer)
    train_featureSet = FeatureSet.pytorch_dataloader(train_loader)
    test_featureSet = FeatureSet.pytorch_dataloader(val_loader)
    estimator.train_minibatch(train_featureSet, zooCriterion, end_trigger=MaxEpoch(args.max_epochs),
                              checkpoint_trigger=EveryEpoch(), validation_set=test_featureSet,
                              validation_method=[Accuracy(), Top5Accuracy()])


if __name__ == '__main__':
    main()

