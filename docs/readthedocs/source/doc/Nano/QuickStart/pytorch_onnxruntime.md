# BigDL-Nano PyTorch ONNXRuntime Acceleration Quickstart

**In this guide we will describe how to apply ONNXRuntime Acceleration on inference pipeline with the APIs delivered by BigDL-Nano**

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

Before you start with onnxruntime accelerator, you are required to install some onnx packages as follows to set up your environment with ONNXRuntime acceleration.
```bash
pip install onnx onnxruntime
```

### **Step 1: Prepare the Model**
```python
import torch
from torchvision import datasets
from torch.utils.data.dataloader import DataLoader
from bigdl.nano.pytorch.vision import transforms
from bigdl.nano.pytorch.trainer import Trainer
from torchvision.models.mobilenetv3 import mobilenet_v3_small
# define your own model
model = mobilenet_v3_small(num_classes=10)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model = Trainer.compile(model, loss, optimizer)
# prepare your own data
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))
])
train_data = datasets.MNIST(
    root=".",
    train=True,
    transform=data_transform
)
data_loader = DataLoader(train_data)
# (Optional) Something else, like training ...
trainer = Trainer()
trainer.fit(model, data_loader)
```

### **Step 2: Apply ONNXRumtime Acceleration**
When you're ready, you can simply append the following part to enable your ONNXRuntime acceleration.
```python
# trace your model as an ONNXRuntime model
# The argument `input_sample` is not required in the following cases:
# you have run `trainer.fit` before trace
# The Model has `example_input_array` set
ort_model = Trainer.trace(model, accelerator='onnxruntime', input_sample=torch.randn((1, 3, 256, 256)))

# The usage is almost the same with any PyTorch module
y_hat = ort_model(torch.rand((10, 3, 256, 256)))
```
- Note
    `ort_model` is not trainable any more, so you can't use like trainer.fit(ort_model, dataloader)