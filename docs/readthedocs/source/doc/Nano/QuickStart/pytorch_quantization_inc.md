# BigDL-Nano PyTorch Quantization with INC Quickstart

**In this guide we will describe how to obtain a quantized model with the APIs delivered by BigDL-Nano**

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

By default, Intel Neural Compressor is not installed with BigDL-Nano. So if you determine to use it as your quantization backend, you'll need to install it first:
```bash
pip install neural-compressor==1.11
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

### **Step 2: Quantization using Intel Neural Compressor**
Quantization is widely used to compress models to a lower precision, which not only reduces the model size but also accelerates inference. BigDL-Nano provides `Trainer.quantize()` API for users to quickly obtain a quantized model with accuracy control by specifying a few arguments.

Without extra accelerator, `Trainer.quantize()` returns a pytorch module with desired precision and accuracy. You can add quantization as below:
```python
q_model = trainer.quantize(model, calib_dataloader=data_loader)
# run simple prediction with transparent acceleration
y_hat = q_model(torch.rand((10, 3, 256, 256)))
```
This is a most basic usage to quantize a model with defaults, INT8 precision, and without search tuning space to control accuracy drop. 
