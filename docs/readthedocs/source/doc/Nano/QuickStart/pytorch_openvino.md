# BigDL-Nano PyTorch OpenVINO Acceleration Quickstart

**In this guide we will describe how to apply OpenVINO Acceleration on inference pipeline with the APIs delivered by BigDL-Nano in 3 simple steps**

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

To use OpenVINO acceleration, you have to install the OpenVINO toolkit:
```bash
pip install openvino-dev
```

### **Step 1: Load the data**
```python
from torchvision.io import read_image
from bigdl.nano.pytorch.vision import transforms

paths = ["../Image/cat.jpg", "../Image/dog.jpg"]
data_transform =  transforms.Compose([transforms.RandomCrop(224),
                                     transforms.RandomHorizontalFlip(p=0.3),
                                     transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
cat = data_transform(read_image(paths[0]))
dog = data_transform(read_image(paths[1]))
```
Let’s have a quick look at our data<br>

<img src="../Image/cat.jpg" width="20%" height="20%" alt="cat" align=center />
<img src="../Image/dog.jpg" width="20%" height="20%" alt="dog" align=center />

### **Step 2: Prepare the Model**
```python
import torch
import torch.nn as nn
from torchvision.models import resnet18
# define your own model
class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18=resnet18(pretrained=True)

    def forward(self, x):
        y_hat = self.resnet18(x)
        return y_hat.argmax(dim=1)
# (Optional) Something else, like training ...
# trainer = Trainer()
# trainer.fit(model, data_loader)
```

### **Step 3: Apply OpenVINO Acceleration**
When you're ready, you can simply append the following part to enable your OpenVINO acceleration.
```python
batch = torch.stack([cat, dog])
predictor = Predictor()
predictor.eval()
predictor(batch)
# trace your model as an OpenVINO model
# The argument `input_sample` is not required in the following cases:
# you have run `trainer.fit` before trace
# The Model has `example_input_array` set
ov_predictor = Trainer.trace(model, accelerator='openvino', input_sample=batch)

# The usage is almost the same with any PyTorch module
y_hat = ov_predictor(batch)
```
- Note
    The `ov_predictor` is not trainable any more, so you can't use like trainer.fit(ov_model, dataloader)