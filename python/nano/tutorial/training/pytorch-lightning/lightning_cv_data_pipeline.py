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

import torch
# from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch
from torchvision.models import resnet18
from bigdl.nano.pytorch import Trainer
from torchmetrics import Accuracy
import pytorch_lightning as pl
from bigdl.nano.pytorch.vision.datasets import ImageFolder
from bigdl.nano.pytorch.vision import transforms
from torchvision.datasets.utils import download_and_extract_archive

DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

class MyLightningModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        # Here the size of each output sample is set to 37.
        self.model.fc = torch.nn.Linear(num_ftrs, 37)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        pred = torch.argmax(output, dim=1)
        acc = torch.sum(y == pred).item() / (len(y) * 1.0)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)


def create_dataloaders(root_dir):

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor()
    ])
    import os
    # Apply data augmentation to the tarin_dataset
    root = os.path.realpath(root_dir)
    dataset = ImageFolder(root=root, transform=data_transform)
    
    # obtain dataset indices that will be used for validation
    dataset_size = len(dataset)
    train_split = 0.8
    train_size = int(dataset_size * train_split)
    val_size = dataset_size - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # prepare data loaders
    train_dataloader = DataLoader(train_set, batch_size=32)
    val_dataloader = DataLoader(val_set, batch_size=32)

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # get dataset
    download_and_extract_archive(url=DATA_URL, download_root="data")
    model = MyLightningModule()
    train_loader, val_loader = create_dataloaders("data")
    # CV Data Pipelines
    #
    # Computer Vision task often needs a data processing pipeline that sometimes constitutes a 
    # non-trivial part of the whole training pipeline. 
    # BigDL-Nano can accelerate computer vision data pipelines.
    # Use ImageFolder and transforms in Dataloader.
    
    trainer = Trainer(max_epochs=5)
    trainer.fit(model, train_dataloaders=train_loader)
    trainer.validate(model, dataloaders=val_loader)