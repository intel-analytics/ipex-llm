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
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data.dataloader import DataLoader
import torch
from torchvision.models import resnet18
from bigdl.nano.pytorch import Trainer
import pytorch_lightning as pl


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
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)


def create_dataloaders():
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(brightness=.5, hue=.3),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    # Apply data augmentation to the tarin_dataset
    train_dataset = OxfordIIITPet(root="/tmp/data", transform=train_transform, download=True)
    val_dataset = OxfordIIITPet(root="/tmp/data", transform=val_transform)

    # obtain training indices that will be used for validation
    indices = torch.randperm(len(train_dataset))
    val_size = len(train_dataset) // 4
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])

    # prepare data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    return train_dataloader, val_dataloader


if __name__ == "__main__":

    model = MyLightningModule()
    train_loader, val_loader = create_dataloaders()

    # Bfloat16 Training
    #
    # BFloat16 is a custom 16-bit floating point format for machine learning
    # that’s comprised of one sign bit, eight exponent bits, and seven mantissa bits.
    # BFloat16 has a greater "dynamic range" than FP16. This means it is able to
    # improve numerical stability than FP16 while delivering increased performance
    # and reducing memory usage.
    #
    # In BigDL-Nano, you can easily enable BFloat16 Mixed precision by setting precision='bf16'
    #
    # Note: Using BFloat16 precision with torch < 1.12 may result in extremely slow training.
    trainer = Trainer(max_epochs=5, precision='bf16')
    trainer.fit(model, train_dataloaders=train_loader)
    trainer.validate(model, dataloaders=val_loader)
    # You can also set use_ipex=True and precision='bf16' to enable ipex optimizer fusion
    # for bf16 to gain more acceleration from BFloat16 data type.
    trainer = Trainer(max_epochs=5, use_ipex=True, precision='bf16')
    trainer.fit(model, train_dataloaders=train_loader)
    trainer.validate(model, dataloaders=val_loader)
