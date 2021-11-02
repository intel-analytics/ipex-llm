# BigDL Nano

BigDL Nano is a python package to transparently accelerate PyTorch and TensorFlow applications on Intel hardware requiring only a few lines of code changes.

## TensorFlow

### Installation

```bash
pip install bigdl-nano[tensorflow]
source bigdl-nano-init
```

### Usage

BigDL Nano currently supports `tf.keras` API in TensorFlow 2.x.

Possible changes are 

- Change `from tensorflow.keras import Sequential` to `from bigdl.nano.tf.keras import Sequential`
- Change `from tensorflow.keras import Model` to `from bigdl.nano.tf.keras import Model`

For example,

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

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

## PyTorch

### Installation

```bash
pip install bigdl-nano[pytorch]
source bigdl-nano-init
```

### Usage

BigDL Nano currently supports (pytorch-lightning)[https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html] API.


Possible changes are 

- Change `from pytorch_lightning import Trainer` to `from bigdl.nano.pytorch import Trainer`

If you are using torchvision, you may also apply the following changes for speed up.
- Change `from torchvision.datasets import ImageFolder` to `from bigdl.nano.pytorch.vision.datasets import ImageFolder`
- Change `from torchvision import transforms` to `from bigdl.nano.pytorch.vision import transforms`

For example,


```python
from bigdl.nano.pytorch.vision.datasets import ImageFolder
from bigdl.nano.pytorch.vision import transforms
from torch.utils.data import DataLoader

data_transform = transforms.Compose([
   transforms.Resize(256),
   transforms.ColorJitter(),
   transforms.RandomCrop(224),
   transforms.RandomHorizontalFlip(),
   transforms.Resize(128),
   transforms.ToTensor()
])
dataset = ImageFolder(args.data_path, transform=data_transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
net = create_model(args.model, args.quantize)
trainer = Trainer(max_epochs=1)
trainer.fit(net, train_loader)
```
