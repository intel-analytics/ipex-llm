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
# ==============================================================================
# Most of the pytorch code is adapted from S0MNATHS's notebook for
# BrainMRI dataset.
# https://www.kaggle.com/s0mnaths/brain-mri-unet-pytorch/notebook
#

from __future__ import print_function
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import albumentations as A
import pandas as pd
import argparse
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.utils import make_grid

from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator

from Unet import UNet
from dataset import *


def dataset(root_path):
    mask_files = glob.glob(root_path + '/*/*_mask*')
    image_files = [file.replace('_mask', '') for file in mask_files]
    files_df = pd.DataFrame({"image_path": image_files,
                             "mask_path": mask_files,
                             "diagnosis": [diagnosis(x) for x in mask_files]})
    train_df, val_df = train_test_split(
        files_df, stratify=files_df['diagnosis'], test_size=0.1, random_state=0)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df, test_df = train_test_split(
        train_df, stratify=train_df['diagnosis'], test_size=0.15, random_state=0)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print("Train: {}\nVal: {}\nTest: {}".format(train_df.shape, val_df.shape, test_df.shape))

    return train_df, val_df, test_df


def dice_coef_metric(pred, label):
    pred[pred >= 0.5] = 1.0
    pred[pred < 0.5] = 0.0
    intersection = 2.0 * (pred * label).sum()
    union = pred.sum() + label.sum()
    if pred.sum() == 0 and label.sum() == 0:
        return 1.
    return intersection / union


def dice_coef_loss(pred, label):
    smooth = 1.0
    intersection = 2.0 * (pred * label).sum() + smooth
    union = pred.sum() + label.sum() + smooth
    return 1 - (intersection / union)


def bce_dice_loss(pred, label):
    dice_loss = dice_coef_loss(pred, label)
    bce_loss = nn.BCELoss()(pred, label)
    return dice_loss + bce_loss


def loss_creator(config):
    return bce_dice_loss


def show_batch(dl):
    for images, masks in dl:
        fig1, ax1 = plt.subplots(figsize=(24, 24))
        ax1.set_xticks([])
        ax1.set_yticks([])
        denorm_images = denormalize(images)
        ax1.imshow(make_grid(denorm_images[:13], nrow=13).permute(1, 2, 0).clamp(0, 1))
        fig2, ax2 = plt.subplots(figsize=(24, 24))
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(make_grid(masks[:13], nrow=13).permute(1, 2, 0).clamp(0, 1))
        break


def train_loader_creator(config, batch_size):
    train_transform = A.Compose([
        A.Resize(width=128, height=128, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
    ])

    train_ds = BrainDataset(config['train'], train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader


def val_loader_creator(config, batch_size):
    val_transform = A.Compose([
        A.Resize(width=128, height=128, p=1.0),
        A.HorizontalFlip(p=0.5),
    ])

    val_ds = BrainDataset(config['val'], val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return val_loader


def test_loader_creator(config, batch_size):
    test_transform = A.Compose([
        A.Resize(width=128, height=128, p=1.0)
    ])

    test_ds = BrainDataset(config['test'], test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader


def model_creator(config=None):
    net = UNet(3, 1)
    return net


def optim_creator(model, config):
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    return optimizer


def scheduler_creator(optimizer, config):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    return scheduler


def denormalize(images):
    means = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    return images * stds + means


parser = argparse.ArgumentParser(description='PyTorch brainMRI Example')
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The cluster mode, such as local, yarn-client, or spark-submit.')
parser.add_argument('--backend', type=str, default="ray",
                    help='The backend of PyTorch Estimator; bigdl, ray, and spark are supported')
parser.add_argument('--batch_size', type=int, default=64, help='The training batch size')
parser.add_argument('--epochs', type=int, default=2, help='The number of epochs to train for')
parser.add_argument('--data_dir', type=str, default='./kaggle_3m', help='The path to the dataset')
parser.add_argument('--additional_archive', type=str, default="kaggle_3m.zip#kaggle_3m",
                    help='The zip dataset for yarn-client mode')
parser.add_argument('--model_dir', type=str, default=os.getcwd(),
                    help="The model save dir for spark backend")
args = parser.parse_args()

if args.cluster_mode == "local":
    init_orca_context(memory='4g')
elif args.cluster_mode.startswith('yarn'):
    init_orca_context(cluster_mode="yarn-client", cores=2,
                      num_nodes=2, additional_archive=args.additional_archive,
                      extra_python_lib='dataset.py,Unet.py', num_executors=2, memory='4g')
elif args.cluster_mode == "spark-submit":
    init_orca_context(cluster_mode="spark-submit")
else:
    invalidInputError(False,
                      "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                      " 'spark-submit', but got " + args.cluster_mode)

train_df, val_df, test_df = dataset(args.data_dir)
batch_size = args.batch_size
epochs = args.epochs
config = {
    'lr': 1e-3,
    'train': train_df,
    'test': test_df,
    'val': val_df
}
train_loader = train_loader_creator(config=config, batch_size=batch_size)


# plot some random training images.
# You should use jupyter notebook to show the images.
show_batch(train_loader)
if args.backend == "bigdl":
    net = model_creator()
    optimizer = optim_creator(model=net, config={"lr": 0.001})
    orca_estimator = Estimator.from_torch(model=net,
                                          optimizer=optimizer,
                                          loss=bce_dice_loss,
                                          config=config,
                                          backend=args.backend)

    orca_estimator.fit(data=train_loader, epochs=args.epochs)


elif args.backend in ["ray", "spark"]:
    orca_estimator = Estimator.from_torch(model=model_creator,
                                          optimizer=optim_creator,
                                          loss=loss_creator,
                                          model_dir=args.model_dir,
                                          backend=args.backend,
                                          metrics=dice_coef_metric,
                                          config=config,
                                          use_tqdm=True,
                                          scheduler_creator=scheduler_creator,
                                          scheduler_step_freq='epoch')

    orca_estimator.fit(data=train_loader_creator, epochs=args.epochs, batch_size=batch_size,
                       validation_data=val_loader_creator)
    res = orca_estimator.evaluate(data=test_loader_creator, batch_size=batch_size)
    for r, value in res.items():
        print(r, ":", value)
else:
    invalidInputError(False, "Only bigdl, ray, and spark are supported as the backend,"
                             " but got {}".format(args.backend))

stop_orca_context()
