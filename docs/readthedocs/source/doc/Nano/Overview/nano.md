# Nano in 5 minutes

BigDL-Nano is a Python package to transparently accelerate PyTorch and TensorFlow applications on Intel hardware. It provides a unified and easy-to-use API for several optimization techniques and tools, so that users can only apply a few lines of code changes to make their PyTorch or TensorFlow code run faster.

----


### PyTorch Bite-sized Example

BigDL-Nano supports both PyTorch and PyTorch Lightning models and most optimizations require only changing a few "import" lines in your code and adding a few flags.

BigDL-Nano uses a extended version of PyTorch Lightning trainer for integrating our optimizations.

For example, if you are using a LightningModule, you can use the following code snippet to enable intel-extension-for-pytorch and multi-instance training.

```python
from bigdl.nano.pytorch import Trainer
net = create_lightning_model()
train_loader = create_training_loader()
trainer = Trainer(max_epochs=1, use_ipex=True, num_processes=4)
trainer.fit(net, train_loader)
```

If you are using custom training loop, you can use the following code to enable intel-extension-for-pytorch, multi-instance training and other nano's optimizations.

```python
from bigdl.nano.pytorch import TorchNano

class MyNano(TorchNano):
    def train(...):
      # copy your train loop here and make a few changes
      ...

MyNano(use_ipex=True, num_processes=2).train()
```

For more details on the BigDL-Nano's PyTorch usage, please refer to the [PyTorch Training](./pytorch_train.md) and [PyTorch Inference](./pytorch_inference.md) page.


### TensorFlow Bite-sized Example

BigDL-Nano supports `tensorflow.keras` API and most optimizations require only changing a few "import" lines in your code and adding a few flags.

BigDL-Nano uses a extended version of `tf.keras.Model` or `tf.keras.Sequential` for integrating our optimizations.

For example, you can conduct a multi-instance training using the following lines of code:

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

For more details on the BigDL-Nano's Tensorflow usage, please refer to the [TensorFlow Training](./tensorflow_train.md) and [TensorFlow Inference](./tensorflow_inference.md) page.
