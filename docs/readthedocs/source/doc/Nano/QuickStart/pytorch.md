# BigDL-Nano PyTorch Overview

BigDL-Nano can be used to accelerate PyTorch or PyTorch-Lightning applications on both training and inference workloads. The optimizations in BigDL-Nano are delivered through a extended version of PyTorch-Lightning `Trainer`. These optimizations are either enabled by default, or can be easily turned on by setting a parameter or calling a method.

## PyTorch Training

We will briefly describe here the major features in BigDL-Nano for PyTorch training. You can find complete examples here [links to be added]().

### Best Known Configurations

When you run `source bigdl-nano-init`, BigDL-Nano will export a few environment variables, such as OMP_NUM_THREADS and KMP_AFFINITY, according to your current hardware. Emprically, these environment variables work best for most PyTorch applications. After setting these environment variables, you can just run your applications as usual (`python app.py`) and no additional changes are required.

### BigDL-Nano PyTorch Trainer

The PyTorch Trainer (`bigdl.nano.pytorch.Trainer`) is the place where we integrate most optimizations. It extends PyTorch Lightning's Trainer and has a few more parameters and methods specific to BigDL-Nano. The Trainer can be directly used to train a `LightningModule`.

For example,

```python
from pytorch_lightning import LightningModule
from bigdl.nano.pytorch import Trainer

class MyModule(LightningModule):
    #  LightningModule definition

from bigdl.nano.pytorch import Trainer
lightning_module = MyModule()
trainer = Trainer(max_epoch=10)
trainer.fit(lightning_module, train_loader)
```

For regulare pytorch modules, we also provides a "compile" methods, that takes in a PyTorch module, an optimizer, and other PyTorch objects and "compile" them into a `LightningModule`.

For example,

```python
from bigdl.nano.pytorch import Trainer
lightning_module = Trainer.compile(pytorch_module, optimizer, scheduler)
trainer = Trainer(max_epoch=10)
trainer.fit(lightning_module, train_loader)
```

#### Intel® Extension for PyTorch

Intel Extension for Pytorch (a.k.a. IPEX) extends PyTorch with optimizations for extra performance boost on Intel hardware. BigDL-Nano integrates IPEX through the `Trainer`. Users can turn on IPEX by setting `use_ipex=True`.

```python
from bigdl.nano.pytorch import Trainer
trainer = Trainer(max_epoch=10, use_ipex=True)
```

#### Multi-instance Training

When training on server with dozens of CPU cores, it is often beneficial to use multiple training instances in a data parallel fashion to make full use the CPU cores. However, using pytorch's DDP API is a little cumbersome and error-prone, and if not configured correctly, it will make the training even slow.

BigDL-Nano makes it very easy to to conduct multi-instance training. You can just set the `num_processes` parameter in the `Trainer` constructor and BigDL-Nano will launch the specific number of processes to perform data parallel training. Each process will be automaticall pinned to a different subset of CPU cores to avoid conflict and maximize training throughput.

```python
from bigdl.nano.pytorch import Trainer
trainer = Trainer(max_epoch=10, num_processes=4)
```

Note that the effective batch size multi-instance training is the `batch_size` in your dataloader times `num_processes` so the number of iterations of each epoch will be reduced `num_processes` fold. A common practice to compensate that is to gradually increase the learning rate to `num_processes` times. You can find more details of this trick in the [facebook paper](https://arxiv.org/abs/1706.02677).

### Optimized Data pipeline

Computer Vision task often needs a data processing pipeline that sometimes consitute a non-trivial part the whole training pipeline. Leveraging OpenCV and libjpeg-turbo, BigDL-Nano can accelerate computer vision data pipelines by providing a drop-in replacement of torch_vision's `datasets` and `transforms`.

```python
from bigdl.nano.pytorch.vision.datasets import ImageFolder
from bigdl.nano.pytorch.vision import transforms

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor()
])

train_set = ImageFolder(train_path, data_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
trainer.fit(module, train_loader)
```

## PyTorch Inference

add link for examples here.

### Runtime Acceleration

onnx runtime, openvino

### Quantization
