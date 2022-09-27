## BigDL-Nano TensorFLow Quantization Quickstart
**In this guide we will demonstrates how to apply post-training quantization on a keras model with BigDL-Nano in 4 simple steps.**

### **Step 0: Prepare Environment**

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](../../UserGuide/python.md) for more details.

```bash
conda create py37 python==3.7.10 setuptools==58.0.4
conda activate py37
# nightly bulit version
pip install --pre --upgrade bigdl-nano[tensorflow]
# set env variables for your conda environment
source bigdl-nano-init
```

By default, [Intel Neural Compressor](https://github.com/intel/neural-compressor) is not installed with BigDL-Nano. So if you determine to use it as your quantization backend, you'll need to install it first:
```bash
pip install neural-compressor==1.11.0
```

BigDL-Nano provides several APIs which can help users easily apply optimizations on inference pipelines to improve latency and throughput. The Keras Model(`bigdl.nano.tf.keras.Model`) and Sequential(`bigdl.nano.tf.keras.Sequential`) provides the APIs for all optimizations you need for inference.

```python
from bigdl.nano.tf.keras import Model, Sequential
```

### Step 1: Loading Data

Here we load data from tensorflow_datasets. The [Imagenette](https://github.com/fastai/imagenette) is a subset of 10 easily classified classes from the Imagenet dataset.

```python
import tensorflow_datasets as tfds
DATANAME = 'imagenette/320px-v2'
(train_ds, test_ds), info = tfds.load(DATANAME, data_dir='../data/',
                                     split=['train', 'validation'],
                                     with_info=True,
                                     as_supervised=True)
```

#### Prepare Inputs
Here we resize the input image to uniform `IMG_SIZE` and the labels are put into one_hot.

```python
import tensorflow as tf
img_size = 224
num_classes = info.features['label'].num_classes
train_ds = train_ds.map(lambda img, label: (tf.image.resize(img, (img_size, img_size)), tf.one_hot(label, num_classes))).batch(32)
test_ds = test_ds.map(lambda img, label: (tf.image.resize(img, (img_size, img_size)), tf.one_hot(label, num_classes))).batch(32)
```

### Step 2: Build Model
Here we initialize the ResNet50 from `tf.keras.applications` with pre-trained ImageNet weights.
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
x = tf.cast(inputs, tf.float32)
x = tf.keras.applications.resnet50.preprocess_input(x)
backbone = ResNet50(weights='imagenet')
backbone.trainable = False
x = backbone(x)
x = layers.Dense(512, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# fit
model.fit(train_ds, epochs=1)
```

### Step 3: Quantization with Intel Neural Compressor
[`Model.quantize()`](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Nano/tensorflow.html#bigdl.nano.tf.keras.Model) return a Keras module with desired precision and accuracy. Taking Resnet50 as an example, you can add quantization as below.

```python
from tensorflow.keras.metrics import CategoricalAccuracy
q_model = model.quantize(calib_dataset=dataset,
                         metric=CategoricalAccuracy(),
                         tuning_strategy='basic'
                         )
```
The quantized model can be called to do inference as normal keras model.
```python
# run simple prediction with transparent acceleration
for img, _ in dataset:
    q_model(img)
```
