# Nano User Guide

## **1. Overview**

BigDL Nano is a python package to transparently accelerate PyTorch and TensorFlow applications on Intel hardware. It provides a unified and easy-to-use API for several optimization techniques and tools, so that users can only apply a few lines of code changes to make their PyTorch or TensorFlow code run faster.

---
## **2. Install**

Note: For windows Users, we recommend using Windows Subsystem for Linux 2 (WSL2) to run BigDL-Nano. Please refer [here](./windows_guide.md) for instructions.

BigDL-Nano can be installed using pip and we recommend installing BigDL-Nano in a conda environment.

For PyTorch Users, you can install bigdl-nano along with some dependencies specific to PyTorch using the following command.

```bash
conda create -n env
conda activate env
pip install bigdl-nano[pytorch]
```

For TensorFlow users, you can install bigdl-nano along with some dependencies specific to TensorFlow using the following command.

```bash
conda create -n env
conda activate env
pip install bigdl-nano[tensorflow]
```

After installing bigdl-nano, you can run the following command to setup a few environment variables. 

```bash
source bigdl-nano-init
```

The `bigdl-nano-init` scripts will export a few environment variable according to your hardware to maximize performance. 

In a conda environment, this will also add this script to `$CONDA_PREFIX/etc/conda/activate.d/`, which will automaticly run when you activate your current environment.

In a pure pip environment, you need to run `source bigdl-nano-init` every time you open a new shell to get optimal performance and run `source bigdl-nano-unset-env` if you want to unset these environment variables.

---

## **3. Get Started**

#### **3.1 PyTorch**

BigDL-Nano supports both PyTorch and PyTorch Lightning models and most optimizations requires only changing a few "import" lines in your code and adding a few flags.

BigDL-Nano uses a extended version of PyTorch Lightning trainer and LightningLite for integrating our optimizations.

For example, if you are using a LightingModule, you can use the following code to enable intel-extension-for-pytorch and multi-instance training.

```python
from bigdl.nano.pytorch import Trainer
net = create_lightning_model()
train_loader = create_training_loader()
trainer = Trainer(max_epochs=1, use_ipex=True, num_processes=4)
trainer.fit(net, train_loader)
```

If you are using LightningLite, you can use the following code to enable intel-extension-for-pytorch and multi-instance training.

```python
from bigdl.nano.pytorch.lite import LightningLite

class Lite(LightningLite):
    def run(...):
      ...

Lite(use_ipex=True, num_processes=2).run()
```

For more details on the BigDL-Nano's PyTorch usage, please refer to the [PyTorch Training](../QuickStart/pytorch_train.md) and [PyTorch Inference](../QuickStart/pytorch_inference.md) page.

### **3.2 TensorFlow**

BigDL-Nano supports `tensorflow.keras` API and most optimizations requires only changing a few "import" lines in your code and adding a few flags.

BigDL-Nano uses a extended version of `tf.keras.Model` or `tf.keras.Sequential` for integrating our optimizations.

For example, you can conduct a multi-instance training using the following code:

```python
import tensorflow as tf
from bigdl.nano.tf.keras import Sequential
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, num_processes=4)
```

For more details on the BigDL-Nano's PyTorch usage, please refer to the [TensorFlow Training](../QuickStart/tensorflow_train.md) and [TensorFlow Inference](../QuickStart/tensorflow_inference.md) page.
