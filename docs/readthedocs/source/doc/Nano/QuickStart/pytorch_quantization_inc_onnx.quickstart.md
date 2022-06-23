# BigDL-Nano PyTorch Quantization with ONNXRuntime accelerator Quickstart

**In this guide we will describe how to obtain a quantized model running inference in the ONNXRuntime engine with the APIs delivered by BigDL-Nano in 2 simple steps**

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

### **Step 1: Prepare Model**
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

### **Step 2: Quantization with ONNXRuntime accelerator**
With the ONNXRuntime accelerator, `Trainer.quantize()` will return a model with compressed precision but running inference in the ONNXRuntime engine.

you can add quantization as below:
```python
ort_q_model = trainer.quantize(model, accelerator='onnxruntime', calib_dataloader=data_loader)

# run simple prediction
y_hat = ort_q_model(torch.rand((10, 3, 224, 224)))
```

Using accelerator='onnxruntime' actually equals to converting the model from Pytorch to ONNX firstly and then do quantization on the converted ONNX model:
```python
ort_model = Trainer.trace(model, accelerator='onnruntime', input_sample=x):
ort_q_model = trainer.quanize(ort_model, accelerator='onnxruntime', calib_dataloader=dataloader)