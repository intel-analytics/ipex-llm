# Adapted from https://www.kaggle.com/s0mnaths/brain-mri-unet-pytorch/notebook
# Data downloaded from https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation

from __future__ import print_function
from UNET import UNet
from dataset import *

import os
from os.path import exists
from os import makedirs
import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import cv2
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torchvision.transforms as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca.learn.trigger import EveryEpoch

import albumentations as A

def dataset():
    ROOT_PATH = '/home/mingxuan/BigDL/python/orca/example/learn/pytorch/brainMRI/data/kaggle_3m/'
    mask_files = glob.glob(ROOT_PATH + '*/*_mask*')
    image_files = [file.replace('_mask', '') for file in mask_files]
    files_df = pd.DataFrame({"image_path": image_files,
                    "mask_path": mask_files,
                    "diagnosis": [diagnosis(x) for x in mask_files]})

    train_df, test_df = train_test_split(files_df, stratify=files_df['diagnosis'], test_size=0.15, random_state=0)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df

train_df, test_df = dataset()

def train_loader_creator(config, batch_size):
    train_transform = A.Compose([
        A.Resize(width=128, height=128, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
    ])
    
    train_ds = BrainDataset(train_df, train_transform)
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    return trainloader


def test_loader_creator(config, batch_size):
    test_transform = A.Compose([
        A.Resize(width=128, height=128, p=1.0)
    ])

    test_ds = BrainDataset(test_df, test_transform)
    testloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    return testloader


def model_creator(config):
    net = UNet(3, 1)
    return net


def optim_creator(model, config):
    optimizer = optim.SGD(model.parameters(),
                          lr=config.get("lr", 0.001),
                          momentum=config.get("momentum", 0.9))
    return optimizer

def denormalize(images):
    means = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    return images * stds + means

def show_batch(dl):
    for images, masks in dl:
        fig1, ax1 = plt.subplots(figsize=(24, 24))
        ax1.set_xticks([]); ax1.set_yticks([])
        denorm_images = denormalize(images)
        ax1.imshow(make_grid(denorm_images[:13], nrow=13).permute(1, 2, 0).clamp(0,1))
        
        fig2, ax2 = plt.subplots(figsize=(24, 24))
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.imshow(make_grid(masks[:13], nrow=13).permute(1, 2, 0).clamp(0,1))
        break
        

parser = argparse.ArgumentParser(description='PyTorch brainMRI Example')
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The cluster mode, such as local, yarn-client, yarn-cluster, spark-submit or k8s.')
parser.add_argument('--backend', type=str, default="bigdl",
                    help='The backend of PyTorch Estimator; '
                        'bigdl, torch_distributed and spark are supported')
parser.add_argument('--batch_size', type=int, default=64, help='The training batch size')
parser.add_argument('--epochs', type=int, default=2, help='The number of epochs to train for')
parser.add_argument('--data_dir', type=str, default="./data", help='The path to dataset')
parser.add_argument('--download', type=bool, default=False, help='Download dataset or not')
parser.add_argument("--executor_memory", type=str, default="5g", help="executor memory")
parser.add_argument("--driver_memory", type=str, default="5g", help="driver memory")
args = parser.parse_args()

if args.cluster_mode == "local":
    init_orca_context(memory="4g")
elif args.cluster_mode.startswith("yarn"):
    if args.cluster_mode == "yarn-client":
        init_orca_context(cluster_mode="yarn-client")
    elif args.cluster_mode == "yarn-cluster":
        init_orca_context(cluster_mode="yarn-cluster", memory=args.executor_memory, driver_memory=args.driver_memory)
elif args.cluster_mode == "spark-submit":
    init_orca_context(cluster_mode="spark-submit")

tensorboard_dir = args.data_dir+"runs"
writer = SummaryWriter(tensorboard_dir + '/brainMRI_experiment_1')

criterion = nn.CrossEntropyLoss()
batch_size = args.batch_size
epochs = args.epochs
root_dir = args.data_dir

train_loader = train_loader_creator(config={"root": root_dir}, batch_size=batch_size)
test_loader = test_loader_creator(config={"root": root_dir}, batch_size=batch_size)

# plot some random training images
# show_batch(train_loader)


if args.backend == "bigdl":
    net = model_creator(config={})
    optimizer = optim_creator(model=net, config={"lr": 0.001})
    orca_estimator = Estimator.from_torch(model=net,
                                        optimizer=optimizer,
                                        loss=criterion,
                                        metrics=[Accuracy()],
                                        backend="bigdl")

    orca_estimator.fit(data=train_loader, epochs=args.epochs, validation_data=test_loader,
                    checkpoint_trigger=EveryEpoch())

    res = orca_estimator.evaluate(data=test_loader)
    print("Accuracy of the network on the test images: %s" % res)
elif args.backend in ["torch_distributed", "spark"]:
    orca_estimator = Estimator.from_torch(model=model_creator,
                                        optimizer=optim_creator,
                                        loss=criterion,
                                        metrics=[Accuracy()],
                                        backend=args.backend,
                                        config={"lr": 0.001,
                                                "root": root_dir},
                                        use_tqdm=True
                                        )

    orca_estimator.fit(data=train_loader_creator, epochs=args.epochs, batch_size=batch_size)

    res = orca_estimator.evaluate(data=test_loader_creator)
    for r in res:
        print(r, ":", res[r])
else:
    raise NotImplementedError("Only bigdl and torch_distributed are supported as the backend,"
                            " but got {}".format(args.backend))

stop_orca_context()

