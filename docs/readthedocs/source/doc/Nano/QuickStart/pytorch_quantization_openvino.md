# BigDL-Nano PyTorch Quantization with POT Quickstart

**In this guide we will describe how to obtain a quantized model with the APIs delivered by BigDL-Nano in 3 simple steps**

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
from torchvision.datasets import OxfordIIITPet
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
data_set = OxfordIIITPet(root="./data/", transform=data_transforms, 
                         target_transform=transforms.Lambda(lambda label: torch.tensor(label, dtype=torch.long)))
data_loader = DataLoader(data_set, batch_size=32, shuffle=True)
```

### **Step 2: Prepare the Model**
```python
from torchvision.models import mobilenet_v3_small
import torch
import torch.nn as nn
# define your own model
model_ft = resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(data_set.classes))
loss_ft = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
from bigdl.nano.pytorch import Trainer
model = Trainer.compile(model_ft, loss_ft, optimizer_ft)
# (Optional) Something else, like training ...
trainer = Trainer(max_epochs=5)
trainer.fit(model, train_dataloader=data_loader)
```

### **Step 3: Quantization using Post-training Optimization Tools**
Accelerator='openvino' means using OpenVINO POT to do quantization. The quantization can be added as below:
```python
from torchmetrics.functional import accuracy
ov_q_model = trainer.quanize(model, accelerator='openvino', calib_dataloader=data_loader, metric=accuracy)

# run simple prediction with transparent acceleration
batch = torch.stack([data_set[0][0], data_set[1][0]])
y_hat = ov_q_model(batch)
```
Same as you set accelerator as ONNXRuntime, it equals to converting the model from Pytorch to OpenVINO firstly and then doing quantization on the converted OpenVINO model:
```python
ov_model = Trainer.trace(model, accelerator='openvino', input_sample=x):
ov_q_model = trainer.quanize(ov_model, accelerator='onnxruntime', calib_dataloader=dataloader)