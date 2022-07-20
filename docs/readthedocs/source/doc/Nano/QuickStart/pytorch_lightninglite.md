# BigDL-Nano Pytorch LightningLite Quickstart

**In this guide we'll demonstrate how to use BigDL-Nano to accelerate custom train loop easily with very few changes.**

### **Step 0: Prepare Environment**

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](../../UserGuide/python.md) for more details.

```bash
conda create py37 python==3.7.10 setuptools==58.0.4
conda activate py37
# nightly bulit version
pip install --pre --upgrade bigdl-nano[pytorch]
# set env variables for your conda environment
source bigdl-nano-init
```

### **Step 1: Load the Data**

Import Cifar10 dataset from torch_vision and modify the train transform. You could access [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) for a view of the whole dataset.

Leveraging OpenCV and libjpeg-turbo, BigDL-Nano can accelerate computer vision data pipelines by providing a drop-in replacement of torch_vision's `datasets` and `transforms`.

```python
from torch.utils.data import DataLoader

from bigdl.nano.pytorch.vision import transforms
from bigdl.nano.pytorch.vision.datasets import CIFAR10

def create_dataloader(data_path, batch_size):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor()
    ])

    train_dataset = CIFAR10(root=data_path, train=True,
                            download=True, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    return train_loader
```

### **Step 2: Define the Model**

You may define your model in the same way as the standard PyTorch models.

```python
from torch import nn

from bigdl.nano.pytorch.vision.models import vision

class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)
```

### Step 3: **Define Train Loop**

Suppose the custom train loop is as follows:

```python
import os
import torch

data_path = os.environ.get("DATA_PATH", ".")
batch_size = 256
max_epochs = 10
lr = 0.01

model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_loader = create_dataloader(data_path, batch_size)

model.train()

for _i in range(max_epochs):
    total_loss, num = 0, 0
    for X, y in train_loader:
        optimizer.zero_grad()
        l = loss(model(X), y)
        l.backward()
        optimizer.step()
        
        total_loss += l.sum()
        num += 1
    print(f'avg_loss: {total_loss / num}')
```

The `LightningLite` (`bigdl.nano.pytorch.lite.LightningLite`) class is the place where we integrate most optimizations. It extends PyTorch Lightning's `LightningLite` class and has a few more parameters and methods specific to BigDL-Nano.

We can accelerate the train loop above by the following steps:

- define a class `Lite` derived from our `LightningLite`
- copy all codes into the `run` method of `Lite`
- add two extra lines to setup model, optimizer and dataloader
- change the backward call

```python
import os
import torch

from bigdl.nano.pytorch.lite import LightningLite

class Lite(LightningLite):
    def run(self):
        # copy all codes into this method
        data_path = os.environ.get("DATA_PATH", ".")
        batch_size = 256
        max_epochs = 10
        lr = 0.01

        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_loader = create_dataloader(data_path, batch_size)

        model, optimizer = self.setup(model, optimizer)      # add this line to setup model and optimizer
        train_loader = self.setup_dataloaders(train_loader)  # add this line to setup dataloader
        model.train()

        for _i in range(max_epochs):
            total_loss, num = 0, 0
            for X, y in train_loader:
                optimizer.zero_grad()
                l = loss(model(X), y)
                self.backward(l)  # change the backward call
                optimizer.step()
                
                total_loss += l.sum()
                num += 1
            print(f'avg_loss: {total_loss / num}')
```

### Step 5: **Run with Nano PyTorch LightningLite**

```python
Lite().run()
```

At this stage, you may already experience some speedup due to the optimized environment variables set by source bigdl-nano-init. Besides, you can also enable optimizations delivered by BigDL-Nano by setting a paramter or calling a method to accelerate PyTorch or PyTorch Lightning application on training workloads.

#### Increase the number of processes in distributed training to accelerate training.

```python
Lite(num_processes=2, strategy="subprocess").run()
```

- Note: BigDL-Nano now support 'spawn', 'subprocess' and 'ray' strategies for distributed training, but only the 'subprocess' strategy can be used in interactive environment.

#### Intel Extension for Pytorch (a.k.a. [IPEX](https://github.com/intel/intel-extension-for-pytorch))

IPEX extends Pytorch with optimizations on intel hardware. BigDL-Nano also integrates IPEX into the `LightningLite`, you can turn on IPEX optimization by setting `use_ipex=True`.

```python
Lite(use_ipex=True, num_processes=2, strategy="subprocess").run()
```
