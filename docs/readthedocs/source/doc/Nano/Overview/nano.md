# Nano User Guide

## **1. Overview**

BigDL Nano is a python package to transparently accelerate PyTorch and TensorFlow applications on Intel hardware. It provides a unified and easy-to-use API for several optimization techniques and tools, so that users can only apply a few lines of code changes to make their PyTorch or TensorFlow code run faster.

---
## **2. Get Started**

### **2.1 PyTorch**

#### **2.1.1 Install**

BigDL-Nano can be installed using pip and we recommend installing BigDL-Nano in a conda environment.

```bash
pip install bigdl-nano[pytorch]
```

After installing bigdl-nano, you can run the following command to setup a few environment variables. 

```bash
source bigdl-nano-init
```

The `bigdl-nano-init` scripts will export a few environment variable according to your hardware to maximize performance. 

In a conda environment, this will also add this script to `$CONDA_PREFIX/etc/conda/activate.d/`, which will automaticly run when you activate your current environment.

In a pure pip environment, you need to run `source bigdl-nano-init` every time you open a new shell to get optimal performance.

#### **2.1.2 Usage**

BigDL-Nano supports both PyTorch and PyTorch Lightning models and most optimizations requires only changing a few "import" lines in your code and adding a few flags.

BigDL-Nano uses a extended version of PyTorch Lightning trainer for integrating our optimizations.

For example, if you are using a LightingModule, you can use the following code enable intel-extension-for-pytorch and multi-instance training.

```python
from bigdl.nano.pytorch import Trainer
net = create_lightning_model()
train_loader = create_training_loader()
trainer = Trainer(max_epochs=1, use_ipex=True, num_processes=4)
trainer.fit(net, train_loader)
```

For more details on the BigDL-Nano's PyTorch usage, please refer to the [PyTorch](./pytorch.md) page.

### **2.2 TensorFlow**

#### **2.2.1 Install**
BigDL-Nano can be installed using pip and we recommend installing BigDL-Nano in a conda environment.

```bash
pip install bigdl-nano[tensorflow]
```

After installing bigdl-nano, you can run the following command to setup a few environment variables. 

```bash
source bigdl-nano-init
```

The `bigdl-nano-init` scripts will export a few environment variable according to your hardware to maximize performance. 

In a conda environment, this will also add this script to `$CONDA_PREFIX/etc/conda/activate.d/`, which will automaticly run when you activate your current environment.

In a pure pip environment, you need to run `source bigdl-nano-init` every time you open a new shell to get optimal performance.


#### **2.2.2 Usage**

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

For more details on the BigDL-Nano's PyTorch usage, please refer to the [TensorFlow](./tensorflow.md) page.
