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
import torchvision
from turbojpeg import TurboJPEG
from typing import Sequence
from typing import Any, Callable, Optional, Union, Tuple
from torch.utils.data.dataloader import DataLoader
from torchvision.models import resnet18
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch.vision.datasets.datasets import local_libturbo_path
import pytorch_lightning as pl
from os.path import split, join, realpath
import cv2
import os
from logging import warning
from PIL import Image
import urllib.request
import os
import stat

LIB_URL = "https://github.com/leonardozcm/libjpeg-turbo/releases/download/2.1.1/libturbojpeg.so.0.2.0"
# These images in the pet dataset that don't have a proper format. 
# Some of them are actually .png files instead .jpg, 
# even though they are in .jpg extension.
SPECIAL_IMAGES = [
    "/tmp/data/oxford-iiit-pet/images/Egyptian_Mau_14.jpg",
    "/tmp/data/oxford-iiit-pet/images/Egyptian_Mau_139.jpg",
    "/tmp/data/oxford-iiit-pet/images/Egyptian_Mau_145.jpg",
    "/tmp/data/oxford-iiit-pet/images/Egyptian_Mau_156.jpg",
    "/tmp/data/oxford-iiit-pet/images/Egyptian_Mau_167.jpg",
    "/tmp/data/oxford-iiit-pet/images/Egyptian_Mau_177.jpg",
    "/tmp/data/oxford-iiit-pet/images/Egyptian_Mau_186.jpg",
    "/tmp/data/oxford-iiit-pet/images/Egyptian_Mau_191.jpg",
    "/tmp/data/oxford-iiit-pet/images/Abyssinian_5.jpg",
    "/tmp/data/oxford-iiit-pet/images/Abyssinian_34.jpg",
    "/tmp/data/oxford-iiit-pet/images/chihuahua_121.jpg",
    "/tmp/data/oxford-iiit-pet/images/beagle_116.jpg",
]
def download_libs(url: str):
    libs_dir = "/tmp/libs"
    if not os.path.exists(libs_dir):
        os.makedirs(libs_dir, exist_ok=True)
    libso_file_name = url.split('/')[-1]
    libso_file = os.path.join(libs_dir, libso_file_name)
    if not os.path.exists(libso_file):
        print('downloading libturbojpeg.so.0.2.0.....')
        urllib.request.urlretrieve(url, libso_file)
    st = os.stat(libso_file)
    os.chmod(libso_file, st.st_mode | stat.S_IEXEC)

_turbo_path = realpath(join(split(realpath(__file__))[0],
                            "/tmp/libs/libturbojpeg.so.0.2.0"))

if not os.path.exists(local_libturbo_path):
    warning("libturbojpeg.so.0 not found in bigdl-nano, try to load from system.")
    download_libs(LIB_URL)
    local_libturbo_path = _turbo_path

class OxfordIIITPet(torchvision.datasets.OxfordIIITPet):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super(OxfordIIITPet, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.jpeg: Optional[TurboJPEG] = None
        
    def _read_image_to_bytes(self, path: str):
        fd = open(path, 'rb')
        img_str = fd.read()
        fd.close()
        return img_str

    def _decode_img_libjpeg_turbo(self, img_str: str):
        if self.jpeg is None:
            self.jpeg = TurboJPEG(lib_path=local_libturbo_path)
        bgr_array = self.jpeg.decode(img_str)
        return bgr_array

    def __getitem__(self, idx: int):
        path = str(self._images[idx])
        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))
        
        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)
            
        if path in SPECIAL_IMAGES:
                img = Image.open(path).convert("RGB")
        else:        
            if path.endswith(".jpg") or path.endswith(".jpeg"):
                # Use turbo-jpg to accelerate baseline JPEG compression and decompression.
                img_str = self._read_image_to_bytes(path)
                img = self._decode_img_libjpeg_turbo(img_str)
            else:
                img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img, target = self.transforms(img, target)

        img = img.numpy()
        return img.astype('float32'), target  
  
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


def create_dataloaders():
    from bigdl.nano.pytorch.vision import transforms
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
    # get dataset
    model = MyLightningModule()
    # BigDL-Nano can accelerate computer vision data pipelines
    # by providing a drop-in replacement of torch_vision’s datasets and transforms
    train_loader, val_loader = create_dataloaders()
    # CV Data Pipelines
    #
    # Computer Vision task often needs a data processing pipeline that sometimes constitutes a 
    # non-trivial part of the whole training pipeline. 
    # BigDL-Nano can accelerate computer vision data pipelines.
    
    trainer = Trainer(max_epochs=5)
    trainer.fit(model, train_dataloaders=train_loader)
    trainer.validate(model, dataloaders=val_loader)