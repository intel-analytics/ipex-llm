# BigDL-Nano PyTorch Quantization with ONNXRuntime accelerator Quickstart

**In this guide we will describe how to obtain a quantized model running inference in the ONNXRuntime engine with the APIs delivered by BigDL-Nano in 3 simple steps**

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

To quantize model using ONNXRuntime as backend, it is required to install Intel Neural Compressor, onnxruntime-extensions as a dependency of INC and some onnx packages as below
```python
pip install neural-compress==1.11
pip install onnx onnxruntime onnxruntime-extensions
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
data_set = OxfordIIITPet(root="./data/", transform=data_transforms)
data_loader = DataLoader(data_set, batch_size=32, shuffle=True)
```

### **Step 2: Prepare your Model**
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

### **Step 3: Quantization with ONNXRuntime accelerator**
With the ONNXRuntime accelerator, `Trainer.quantize()` will return a model with compressed precision but running inference in the ONNXRuntime engine.

you can add quantization as below:
```python
from torchmetrics.functional import accuracy
ort_q_model = trainer.quantize(model, accelerator='onnxruntime', calib_dataloader=data_loader, metric=accuracy)

# run simple prediction
batch = torch.stack([data_set[0][0], data_set[1][0]])
ort_q_model(batch)
```

Using accelerator='onnxruntime' actually equals to converting the model from Pytorch to ONNX firstly and then do quantization on the converted ONNX model:
```python
ort_model = Trainer.trace(model, accelerator='onnruntime', input_sample=x):
ort_q_model = trainer.quanize(ort_model, accelerator='onnxruntime', calib_dataloader=dataloader)
```