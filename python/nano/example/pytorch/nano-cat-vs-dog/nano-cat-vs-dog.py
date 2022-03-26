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

"""Computer vision example on Image Classfication via Resnet50.
This computer vision example illustrates how one could use BigDL nano to easily train 
a Resnet50 model. For the sake of this example, the 'cats and dogs dataset' should be 
pre-downloaded (follow README.md) and the proposed network is trained for 1 epoch. After 
training, it will output the test result.
"""
import os
import torch
from bigdl.nano.pytorch.vision.datasets import ImageFolder
from bigdl.nano.pytorch.vision.transforms import transforms
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_and_extract_archive
from bigdl.nano.pytorch.vision.models import resnet50 
from bigdl.nano.pytorch.vision.models import ImageClassifier 
from bigdl.nano.common import init_nano 
from bigdl.nano.pytorch.trainer import Trainer

DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

parser = argparse.ArgumentParser(description='PyTorch Cat Dog')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--root_dir', default=None, help='path to cat vs dog dataset'
                    'which should have two folders `cat` and `dog`, each containing'
                    'cat and dog pictures.')


def create_data_loader(root_dir, batch_size):
    dir_path = os.path.realpath(root_dir)
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor()
    ])

    catdogs = ImageFolder(dir_path, data_transform)
    dataset_size = len(catdogs)
    train_split = 0.8
    train_size = int(dataset_size * train_split)
    val_size = dataset_size - train_size
    train_set, val_set = torch.utils.data.random_split(catdogs, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    size = (train_size, val_size)
    return train_loader, val_loader, size


class Classifier(ImageClassifier):

    def __init__(self):
        backbone = resnet50(pretrained=True, include_top=False, freeze=True)
        super().__init__(backbone=backbone, num_classes=2)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002, amsgrad=True)


def main():
    args = parser.parse_args()
    init_nano()
    if args.root_dir is None:
        root_path = "data"
        download_and_extract_archive(url=DATA_URL,download_root="data",remove_finished=True)
    else:
        root_path = args.root_dir

    train_loader, val_loader, _ = create_data_loader(root_path, args.batch_size)
    classifier = Classifier()
    trainer = Trainer(max_epochs=1)
    trainer.fit(classifier, train_loader)
    trainer.test(classifier, val_loader)


if __name__ == "__main__":
    main()
