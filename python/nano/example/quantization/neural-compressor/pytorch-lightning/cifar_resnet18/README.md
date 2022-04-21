This example shows the usage of pytorch quantization based on Intel Neural Compressor in Nano. We mainly focus on post-training static quantization here considering best performance for Convolution-based networks. Code is adapted from Pytorch-Lightning Tutorual [CIFAR10-BaseLine](https://github.com/PyTorchLightning/lightning-tutorials/blob/main/lightning_examples/cifar10-baseline/baseline.py).


## Environment Setup
```shell
pip install bigdl-nano[pytorch]
pip install lightning-bolts
```

## Run
```
python resnet18_cifar.py
```
