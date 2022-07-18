# BigDL-Nano PyTorch Training Overview

BigDL-Nano can be used to accelerate PyTorch or PyTorch-Lightning applications on training workloads. The optimizations in BigDL-Nano are delivered through an extended version of PyTorch-Lightning `Trainer`. These optimizations are either enabled by default or can be easily turned on by setting a parameter or calling a method.

We will briefly describe here the major features in BigDL-Nano for PyTorch training. You can find complete examples [here](https://github.com/intel-analytics/BigDL/tree/main/python/nano/notebooks/pytorch).

### Best Known Configurations

When you run `source bigdl-nano-init`, BigDL-Nano will export a few environment variables, such as OMP_NUM_THREADS and KMP_AFFINITY, according to your current hardware. Empirically, these environment variables work best for most PyTorch applications. After setting these environment variables, you can just run your applications as usual (`python app.py`) and no additional changes are required.

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

For regular PyTorch modules, we also provide a "compile" method, that takes in a PyTorch module, an optimizer, and other PyTorch objects and "compiles" them into a `LightningModule`.

For example,

```python
from bigdl.nano.pytorch import Trainer
lightning_module = Trainer.compile(pytorch_module, loss, optimizer, scheduler)
trainer = Trainer(max_epoch=10)
trainer.fit(lightning_module, train_loader)
```

#### Intel® Extension for PyTorch

Intel Extension for Pytorch (a.k.a. IPEX) [link](https://github.com/intel/intel-extension-for-pytorch) extends PyTorch with optimizations for an extra performance boost on Intel hardware. BigDL-Nano integrates IPEX through the `Trainer`. Users can turn on IPEX by setting `use_ipex=True`.

```python
from bigdl.nano.pytorch import Trainer
trainer = Trainer(max_epoch=10, use_ipex=True)
```

#### Multi-instance Training

When training on a server with dozens of CPU cores, it is often beneficial to use multiple training instances in a data-parallel fashion to make full use of the CPU cores. However, using PyTorch's DDP API is a little cumbersome and error-prone, and if not configured correctly, it will make the training even slow.

BigDL-Nano makes it very easy to conduct multi-instance training. You can just set the `num_processes` parameter in the `Trainer` constructor and BigDL-Nano will launch the specific number of processes to perform data-parallel training. Each process will be automatically pinned to a different subset of CPU cores to avoid conflict and maximize training throughput.

```python
from bigdl.nano.pytorch import Trainer
trainer = Trainer(max_epoch=10, num_processes=4)
```

Note that the effective batch size multi-instance training is the `batch_size` in your `dataloader` times `num_processes` so the number of iterations of each epoch will be reduced `num_processes` fold. A common practice to compensate for that is to gradually increase the learning rate to `num_processes` times. You can find more details of this trick in the [Facebook paper](https://arxiv.org/abs/1706.02677).

### BigDL-Nano PyTorch LightningLite

The `LightningLite` (`bigdl.nano.pytorch.lite.LightningLite`) class is the place where we integrate most optimizations. It extends PyTorch Lightning's `LightningLite` class and has a few more parameters and methods specific to BigDL-Nano.

By using it, we only need to make very few changes to accelerate custom train loop. For example,

```python
from bigdl.nano.pytorch.lite import LightningLite

class Lite(LightningLite) :
    def run(self, ...):
        # copy your train loop here and make a few changes

Lite().run(...)
```

- note: see [this tutorial](./pytorch_lightninglite.html) for details about our `LightningLite`.

Our `LightningLite` also integrates IPEX and distributed training optimizations. For example,

```python
from bigdl.nano.pytorch.lite import LightningLite

class Lite(LightningLite):
    def run(self, ...):
        # define train loop

# enable IPEX optimizaiton
Lite(use_ipex=True).run(...)

# enable IPEX and distributed training, using spawn strategy
Lite(use_ipex=True, num_processes=2, strategy="spawn")
```

### Optimized Data pipeline

Computer Vision task often needs a data processing pipeline that sometimes constitutes a non-trivial part of the whole training pipeline. Leveraging OpenCV and libjpeg-turbo, BigDL-Nano can accelerate computer vision data pipelines by providing a drop-in replacement of torch_vision's `datasets` and `transforms`.

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
