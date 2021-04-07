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
# ==============================================================================
# This example trains a super-resolution network on the BSD300 dataset,
# using crops from the 200 training images, and evaluating on crops of the 100 test images,
# and is adapted from
# https://github.com/pytorch/examples/tree/master/super_resolution
#

from __future__ import print_function
import argparse

from math import log10
from PIL import Image
import urllib
import tarfile
from os import makedirs, remove, listdir
from os.path import exists, join, basename

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.pytorch import Estimator
from zoo.orca.learn.metrics import MSE
from zoo.orca.learn.trigger import EveryEpoch

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int,
                    default=3, help="super resolution upscale factor")
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=10, help='testing batch size')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=4,
                    help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--cluster_mode', type=str,
                    default='local', help='The mode of spark cluster.')
parser.add_argument('--backend', type=str, default="bigdl",
                    help='The backend of PyTorch Estimator; '
                         'bigdl and torch_distributed are supported.')
opt = parser.parse_args()

print(opt)

if opt.cluster_mode == "local":
    init_orca_context()
elif opt.cluster_mode == "yarn":
    additional = None if not exists("dataset/BSDS300.zip") else "dataset/BSDS300.zip#dataset"
    init_orca_context(cluster_mode="yarn-client", cores=4, num_nodes=2,
                      additional_archive=additional)
else:
    print("init_orca_context failed. cluster_mode should be either 'local' or 'yarn' but got "
          + opt.cluster_mode)


def download_report(count, block_size, total_size):
    downloaded = count * block_size
    percent = 100. * downloaded / total_size
    percent = min(100, percent)
    print('downloaded %d, %.2f%% completed' % (downloaded, percent))


def download_bsd300(dest="./dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        file_path = join(dest, basename(url))
        urllib.request.urlretrieve(url, file_path, download_report)

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x)
                                for x in listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.image_filenames)


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def train_data_creator(config, batch_size):
    def get_training_set(upscale_factor):
        root_dir = download_bsd300()
        train_dir = join(root_dir, "train")
        crop_size = calculate_valid_crop_size(256, upscale_factor)

        return DatasetFromFolder(train_dir,
                                 input_transform=input_transform(crop_size, upscale_factor),
                                 target_transform=target_transform(crop_size))

    train_set = get_training_set(config.get("upscale_factor", 3))
    training_data_loader = DataLoader(dataset=train_set,
                                      batch_size=batch_size,
                                      num_workers=config.get("threads", 4),
                                      shuffle=True)
    return training_data_loader


def validation_data_creator(config, batch_size):
    def get_test_set(upscale_factor):
        root_dir = download_bsd300()
        test_dir = join(root_dir, "test")
        crop_size = calculate_valid_crop_size(256, upscale_factor)

        return DatasetFromFolder(test_dir,
                                 input_transform=input_transform(crop_size, upscale_factor),
                                 target_transform=target_transform(crop_size))

    test_set = get_test_set(config.get("upscale_factor", 3))
    testing_data_loader = DataLoader(dataset=test_set,
                                     batch_size=batch_size,
                                     num_workers=config.get("threads", 4),
                                     shuffle=False)
    return testing_data_loader


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


def model_creator(config):
    torch.manual_seed(config.get("seed", 123))
    net = Net(upscale_factor=config.get("upscale_factor", 3))
    return net


def optim_creator(model, config):
    return optim.Adam(model.parameters(), lr=config.get("lr", 0.01))


criterion = nn.MSELoss()
model_dir = "models"

if opt.backend == "bigdl":
    model = model_creator(
        config={
            "upscale_factor": opt.upscale_factor,
            "seed": opt.seed
        }
    )
    optimizer = optim_creator(model, config={"lr": opt.lr})
    estimator = Estimator.from_torch(
        model=model,
        optimizer=optimizer,
        loss=criterion,
        metrics=[MSE()],
        model_dir=model_dir,
        backend="bigdl"
    )

    train_loader = train_data_creator(
        config={
            "upscale_factor": opt.upscale_factor,
            "threads": opt.threads
        },
        batch_size=opt.batch_size
    )
    test_loader = validation_data_creator(
        config={
            "upscale_factor": opt.upscale_factor,
            "threads": opt.threads
        },
        batch_size=opt.batch_size
    )

    estimator.fit(data=train_loader, epochs=opt.epochs, validation_data=test_loader,
                  checkpoint_trigger=EveryEpoch())

    val_stats = estimator.evaluate(data=test_loader)
    print("===> Validation Complete: Avg. PSNR: {:.4f} dB, Avg. Loss: {:.4f}"
          .format(10 * log10(1. / val_stats["MSE"]), val_stats["MSE"]))

elif opt.backend == "torch_distributed":
    estimator = Estimator.from_torch(
        model=model_creator,
        optimizer=optim_creator,
        loss=criterion,
        backend="torch_distributed",
        config={
            "lr": opt.lr,
            "upscale_factor": opt.upscale_factor,
            "threads": opt.threads,
            "seed": opt.seed
        }
    )

    if not exists(model_dir):
        makedirs(model_dir)

    for epoch in range(1, opt.epochs + 1):
        stats = estimator.fit(data=train_data_creator, epochs=1, batch_size=opt.batch_size)
        for epochinfo in stats:
            print("===> Epoch {} Complete: Avg. Loss: {:.4f}"
                  .format(epoch, epochinfo["train_loss"]))

        val_stats = estimator.evaluate(data=validation_data_creator,
                                       batch_size=opt.test_batch_size)
        print("===> Validation Complete: Avg. PSNR: {:.4f} dB, Avg. Loss: {:.4f}"
              .format(10 * log10(1. / val_stats["val_loss"]), val_stats["val_loss"]))

        model_out_path = model_dir + "/" + "model_epoch_{}.pth".format(epoch)
        model = estimator.get_model()
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
else:
    raise NotImplementedError("Only bigdl and torch_distributed are supported as the backend, "
                              "but got {}".format(opt.backend))

stop_orca_context()
