# PyTorch Training

BigDL-Nano can be used to accelerate **PyTorch** or **PyTorch-Lightning** applications on training workloads. These optimizations are either enabled by default or can be easily turned on by setting a parameter or calling a method.

The optimizations in BigDL-Nano are delivered through

1) An extended version of PyTorch-Lightning `Trainer` (`bigdl.nano.pytorch.Trainer`) for `LightningModule` and easy `nn.Module`.

2) An abstract `TorchNano` (`bigdl.nano.pytorch.TorchNano`) or a decorator `@nano` (`bigdl.nano.pytorch.nano`) to accelerate raw or complex `nn.Module`.

We will briefly describe here the major features in BigDL-Nano for PyTorch training. You can find complete how to guides for acceleration of [PyTorch-Lightning](../Howto/index.html#pytorch-lightning) and [PyTorch](../Howto/index.html#pytorch).

## Best Known Environment Variables
When you successfully installed `bigdl-nano` (please refer to [installation guide](./install.html)) in a conda environment. You are **highly recommeneded** to run following command **once**.
```bash
source bigdl-nano-init
```
BigDL-Nano will export a few environment variables, such as `OMP_NUM_THREADS` and `KMP_AFFINITY`, according to your current hardware. Empirically, these environment variables work best for most PyTorch applications. After setting these environment variables, you can just run your applications as usual (e.g., `python app.py` or `jupyter notebook`).

## Accelerate `nn.Module`'s training
`nn.Module` is the abstraction used in PyTorch for AI Model. It's common that users' model is easy enough to be handled by a regular training loop. In other cases, users may have highly customized training loop. Nano could support the acceleration for both cases.

### `nn.Module` with regular training loop
Most of the AI model defined in `nn.Module` could be trained in a similar regular training loop. Any `nn.Module` that 
- Have only one output
- Need only 1 loss function and 1 optimizer (e.g., GAN might not applied)
- Have no special customized checkpoint/evaluation logic

could use `Trainer.compile` that takes in a PyTorch module, a loss, an optimizer, and other PyTorch objects and "compiles" them into a `LightningModule`. And then a `Trainer` instance could be used to train this compiled model.

For example,

```python
from bigdl.nano.pytorch import Trainer

lightning_module = Trainer.compile(pytorch_module, loss, optimizer)
trainer = Trainer(max_epochs=10)
trainer.fit(lightning_module, train_loader)
```

`trainer.fit` will apply all the acceleration methods that could generally be applied to any models. While there are some optional acceleration method for which you could easily enable.

### `nn.Module` with customized training loop

The `TorchNano` class is what we use to accelerate raw PyTorch code. By using it, we only need to make very few changes to accelerate custom training loop. For example,

```python
from bigdl.nano.pytorch import TorchNano

class MyNano(TorchNano) :
    def train(self, ...):
        # copy your train loop here and make a few changes
        ...

MyNano().train(...)
```

```eval_rst
.. important::
   
   Please refer to `here <../Howto/Training/PyTorch/convert_pytorch_training_torchnano.html#Convert-to-TorchNano>`_ for a detailed guide about how to make changes to your custom PyTorch training loop so that you could use ``TorchNano``.

   Note that the most important change is to use ``self.setup`` method to set up model, optimizer(s), and dataloader(s) for acceleration inside the training loop.
```

If you have already defined a PyTorch training loop function with a model, optimizers, and dataloaders as parameters, you could use `@nano` decorator to gain acceleration in a simpler way. For example:

```python
from bigdl.nano.pytorch import nano

@nano()
def train(model, optimizer, train_loader, ...):
    ...

train(model, optimizer, train_loader, ...)
```

```eval_rst
.. seealso::
   
   Please refer to `here <../Howto/Training/PyTorch/use_nano_decorator_pytorch_training.html>`_ for a detailed usage of ``@nano`` decorator.
```

## Accelerate `LightningModule`'s training

The PyTorch `Trainer` extends PyTorch Lightning's `Trainer` and has a few more parameters and methods specific to BigDL-Nano. The Trainer can be directly used to train a `LightningModule`.

For example,

```python
from pytorch_lightning import LightningModule
from bigdl.nano.pytorch import Trainer

class MyModule(LightningModule):
    #  LightningModule definition

lightning_module = MyModule()
trainer = Trainer(max_epochs=10)
trainer.fit(lightning_module, train_loader)
```

## Optional Acceleration Methods
### Intel® Extension for PyTorch*

[Intel Extension for PyTorch*](https://github.com/intel/intel-extension-for-pytorch) (a.k.a. IPEX) extends PyTorch with optimizations for an extra performance boost on Intel hardware.

BigDL-Nano integrates IPEX in `Trainer`, `TorchNano` and `@nano` decorator. Users can turn on IPEX by setting `use_ipex=True`.

```eval_rst

.. tabs::

    .. tab:: Trainer

        .. code-block:: python

            from bigdl.nano.pytorch import Trainer

            trainer = Trainer(max_epochs=10, use_ipex=True)
            trainer.fit(...)

    .. tab:: TorchNano

        .. code-block:: python

            from bigdl.nano.pytorch import TorchNano

            class MyNano(TorchNano) :
                def train(self, ...):
                    # copy your train loop here and make a few changes

            MyNano(use_ipex=True).train(...)

    .. tab:: @nano

        .. code-block:: python

            from bigdl.nano.pytorch import nano

            @nano(use_ipex=True)
            def train(model, optimizer, train_loader, ...):
                ...

            train(model, optimizer, train_loader, ...)

```

### Multi-instance Training

When training on a server with dozens of CPU cores, it is often beneficial to use multiple training instances in a data-parallel fashion to make full use of the CPU cores. However, using PyTorch's DDP API is a little cumbersome and error-prone, and if not configured correctly, it will make the training even slow.

You can just set the `num_processes` parameter in the `Trainer` or `TorchNano` constructor, or the `@nano` decorator, so that BigDL-Nano will launch the specific number of processes to perform data-parallel training. Each process will be automatically pinned to a different subset of CPU cores to avoid conflict and maximize training throughput.

```eval_rst

.. tabs::

    .. tab:: Trainer

        .. code-block:: python

            from bigdl.nano.pytorch import Trainer

            trainer = Trainer(max_epochs=10, num_processes=4)
            trainer.fit(...)

    .. tab:: TorchNano

        .. code-block:: python

            from bigdl.nano.pytorch import TorchNano

            class MyNano(TorchNano) :
                def train(self, ...):
                    # copy your train loop here and make a few changes

            MyNano(num_processes=4).train(...)

    .. tab:: @nano

        .. code-block:: python

            from bigdl.nano.pytorch import nano

            @nano(num_processes=4)
            def train(model, optimizer, train_loader, ...):
                ...

            train(model, optimizer, train_loader, ...)

```
```eval_rst
.. note::
   
   The effective batch size in multi-instance training is the ``batch_size`` in your ``dataloader`` times ``num_processes``. So, the number of iterations of each epoch will be reduced ``num_processes`` fold. To achieve the same effect as single instance training, a common practice to compensate is to gradually increase the learning rate to ``num_processes`` times. 

   BigDL-Nano supports this practice by default through ``auto_lr=Ture``, which will scale the learning rate linearly by ``num_processes`` times.

   To get more details about this 'learning rate warmup' trick, you could refer to `this paper <https://arxiv.org/abs/1706.02677>`_ published by Facebook.
```

### BFloat16 Mixed Precision
BFloat16 Mixed Precison combines BFloat16 and FP32 during training, which could lead to increased performance and reduced memory usage. Compared to FP16 mixed precison, BFloat16 mixed precision has better numerical stability.

You could instantiate a BigDL-Nano `Trainer` or `TorchNano` with `precision='bf16'`, or set `precision='bf16'` in the `@nano` decorator to use BFloat16 mixed precision for training.


```eval_rst

.. tabs::

    .. tab:: Trainer

        .. code-block:: python

            from bigdl.nano.pytorch import Trainer

            trainer = Trainer(max_epochs=5, precision='bf16')
            trainer.fit(...)

    .. tab:: TorchNano

        .. code-block:: python

            from bigdl.nano.pytorch import TorchNano

            class MyNano(TorchNano) :
                def train(self, ...):
                    # copy your train loop here and make a few changes

            MyNano(precision='bf16').train(...)

    .. tab:: @nano

        .. code-block:: python

            from bigdl.nano.pytorch import nano

            @nano(precision='bf16')
            def train(model, optimizer, train_loader, ...):
                ...

            train(model, optimizer, train_loader, ...)

```

### Channels Last Memory Format
You could instantiate a BigDL-Nano `Trainer` or `TorchNano` with `channels_last=True`, or set `channels_last=True` in the `@nano` decorator to use the channels last memory format, i.e. NHWC (batch size, height, width, channels), as an alternative way to store tensors in classic/contiguous NCHW order.

```eval_rst

.. tabs::

    .. tab:: Trainer

        .. code-block:: python

            from bigdl.nano.pytorch import Trainer

            trainer = Trainer(max_epochs=5, channels_last=True)
            trainer.fit(...)

    .. tab:: TorchNano

        .. code-block:: python

            from bigdl.nano.pytorch import TorchNano

            class MyNano(TorchNano) :
                def train(self, ...):
                    # copy your train loop here and make a few changes

            MyNano(channels_last=True).train(...)

    .. tab:: @nano

        .. code-block:: python

            from bigdl.nano.pytorch import nano

            @nano(channels_last=True)
            def train(model, optimizer, train_loader, ...):
                ...

            train(model, optimizer, train_loader, ...)

```


## Accelerate `torchvision` data processing

Computer Vision task often needs a data processing pipeline that sometimes constitutes a non-trivial part of the whole training pipeline.

Leveraging OpenCV and libjpeg-turbo, BigDL-Nano can accelerate computer vision data pipelines by providing a drop-in replacement of `torchvision`'s components such as `datasets` and `transforms`. Nano provides a patch API `patch_torch` to accelerate these functions.


```python
from bigdl.nano.pytorch import patch_torch
patch_torch()

from torchvision.datasets import ImageFolder
from torchvision import transforms

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
```
