# BigDL-Nano PyTorch Quantization with POT Quickstart

**In this guide we will describe how to obtain a quantized model with the APIs delivered by BigDL-Nano in 4 simple steps**

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

The POT(Post-training Optimization Tools) is provided by OpenVINO toolkit. To use POT, you need to install OpenVINO
```python
pip install openvino-dev
```

### **Step 1: Load the data**
```python
import torch
from torchvision.io import read_image
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data.dataloader import DataLoader

train_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ColorJitter(brightness=.5, hue=.3),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
val_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# Apply data augmentation to the tarin_dataset
train_dataset = OxfordIIITPet(root = ".",
                              transform=train_transform,
                              target_transform=transforms.Lambda(lambda label: torch.tensor(label, dtype=torch.long)))   # Quantization using POT expect a tensor as label
val_dataset = OxfordIIITPet(root=".", transform=val_transform)
# obtain training indices that will be used for validation
indices = torch.randperm(len(train_dataset))
val_size = len(train_dataset) // 4
train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])
# prepare data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32)
```

### **Step 2: Prepare the Model**
```python
import torch
from torchvision.models import resnet18
from bigdl.nano.pytorch import Trainer
model_ft = resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 37.
model_ft.fc = torch.nn.Linear(num_ftrs, 37)
loss_ft = torch.nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Compile our model with loss function, optimizer.
model = Trainer.compile(model_ft, loss_ft, optimizer_ft)
trainer = Trainer(max_epochs=5)
trainer.fit(model, train_dataloader=train_dataloader)

# Inference/Prediction
x = torch.stack([val_dataset[0][0], val_dataset[1][0]])
model_ft.eval()
y_hat = model_ft(x)
y_hat.argmax(dim=1)
```

### **Step 3: Quantization using Post-training Optimization Tools**
Accelerator='openvino' means using OpenVINO POT to do quantization. The quantization can be added as below:
```python
from torchmetrics import Accuracy
ov_q_model = trainer.quantize(model, accelerator="openvino", calib_dataloader=data_loader)

# run simple prediction
batch = torch.stack([data_set[0][0], data_set[1][0]])
ov_q_model(batch)
```