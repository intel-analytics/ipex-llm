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
from __future__ import print_function
import argparse
import torch
import torchvision
from torchvision import datasets, transforms
from bigdl.optim.optimizer import *
from zoo.pipeline.api.torch import TorchModel, TorchLoss
from zoo.pipeline.estimator import *
from zoo.common.nncontext import *
from zoo.feature.common import FeatureSet
from zoo.pipeline.api.keras.metrics import Accuracy, Top5Accuracy
from zoo.util.utils import detect_conda_env_name

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
    args = parser.parse_args()
    if os.environ.get('HADOOP_CONF_DIR') is None:
        sc = init_spark_on_local(cores=args.cores, conf={"spark.driver.memory": "20g"})
    else:
        hadoop_conf_dir = os.environ.get('HADOOP_CONF_DIR')
        num_executors = args.nodes
        executor_memory = args.executor_memory
        driver_memory = args.driver_memory
        driver_cores = args.driver_cores
        num_cores_per_executor = args.cores
        os.environ['ZOO_MKL_NUMTHREADS'] = str(num_cores_per_executor)
        os.environ['OMP_NUM_THREADS'] = str(num_cores_per_executor)
        sc = init_spark_on_yarn(
            hadoop_conf=hadoop_conf_dir,
            conda_name=detect_conda_env_name(),  # auto detect current conda env name
            num_executors=num_executors,
            executor_cores=num_cores_per_executor,
            executor_memory=executor_memory,
            driver_memory=driver_memory,
            driver_cores=driver_cores,
            conf={"spark.rpc.message.maxSize": "1024",
                  "spark.task.maxFailures": "1",
                  "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
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

    iterationPerEpoch = int(math.ceil(float(1281167) / args.batch_size))
    step = Step(iterationPerEpoch * 30, 0.1)
    zooOptimizer = SGD(args.lr, momentum=args.momentum, dampening=0.0,
                       leaningrate_schedule=step, weightdecay=args.weight_decay)
    zooModel = TorchModel.from_pytorch(model)
    criterion = torch.nn.CrossEntropyLoss()
    zooCriterion = TorchLoss.from_pytorch(criterion)
    estimator = Estimator(zooModel, optim_methods=zooOptimizer)
    train_featureSet = FeatureSet.pytorch_dataloader(train_loader)
    test_featureSet = FeatureSet.pytorch_dataloader(val_loader)
    estimator.train_minibatch(train_featureSet, zooCriterion, end_trigger=MaxEpoch(90),
                              checkpoint_trigger=EveryEpoch(), validation_set=test_featureSet,
                              validation_method=[Accuracy(), Top5Accuracy()])


if __name__ == '__main__':
    main()
