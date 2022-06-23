# BigDL-Nano PyTorch Quantization with POT Quickstart

**In this guide we will describe how to obtain a quantized model with the APIs delivered by BigDL-Nano in 2 simple steps**

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

### **Step 1: Prepare the Model**
```python
import torch
from torchvision import datasets
from torch.utils.data.dataloader import DataLoader
from bigdl.nano.pytorch.vision import transforms
from bigdl.nano.pytorch.trainer import Trainer
from torchvision.models import resnet18
# define your own model
model = resnet18(num_classes=10)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model = Trainer.compile(model, loss, optimizer)
# prepare your own data
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

### **Step 2: Quantization using Post-training Optimization Tools**
Accelerator='openvino' means using OpenVINO POT to do quantization. The quantization can be added as below:
```python
ov_q_model = trainer.quanize(model, accelerator='openvino', calib_dataloader=data_loader)

# run simple prediction with transparent acceleration
y_hat = ov_q_model(torch.rand((10, 3, 224, 224)))
```
Same as you set accelerator as ONNXRuntime, it equals to converting the model from Pytorch to OpenVINO firstly and then doing quantization on the converted OpenVINO model:
```python
ov_model = Trainer.trace(model, accelerator='openvino', input_sample=x):
ov_q_model = trainer.quanize(ov_model, accelerator='onnxruntime', calib_dataloader=dataloader)