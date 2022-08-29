# BigDL-Nano TensorFlow Training Quickstart
**In this guide we will describe how to accelerate TensorFlow Keras application on training workloads using BigDL-Nano in 5 simple steps**

### **Step 0: Prepare Environment**

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](../../UserGuide/python.md) for more details.

```bash
conda create py37 python==3.7.10 setuptools==58.0.4
conda activate py37
# nightly bulit version
pip install --pre --upgrade bigdl-nano[tensorflow]
# set env variables for your conda environment
source bigdl-nano-init
pip install tensorflow-datasets
```

### **Step 1: Import BigDL-Nano**
The optimizations in BigDL-Nano are delivered through BigDL-Nano’s `Model` and `Sequential` classes. For most cases, you can just replace your `tf.keras.Model` to `bigdl.nano.tf.keras.Model` and `tf.keras.Sequential` to `bigdl.nano.tf.keras.Sequential` to benefits from BigDL-Nano.
```python
from bigdl.nano.tf.keras import Model, Sequential
```

### **Step 2: Load the Data**
Here we load data from tensorflow_datasets(hereafter [TFDS](https://www.tensorflow.org/datasets)). The [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html) dataset contains images of 120 breeds of dogs around the world. There are 20,580 images, out of which 12,000 are used for training and 8580 for testing.
```python
import tensorflow_datasets as tfds
(ds_train, ds_test), ds_info = tfds.load(
    "stanford_dogs",
    data_dir="../data/",
    split=['train', 'test'],
    with_info=True,
    as_supervised=True
)
```
#### Prepare Inputs
When the dataset include images with various size, we need to resize them into a shared size. The labels are put into one-hot. The dataset is batched.
```python
import tensorflow as tf
img_size = 224
num_classes = ds_info.features['label'].num_classes
batch_size = 64
def preprocessing(img, label):
    return tf.image.resize(img, (img_size, img_size)), tf.one_hot(label, num_classes)
AUTOTUNE = tf.data.AUTOTUNE
ds_train = ds_train.cache().repeat().shuffle(1000).map(preprocessing).batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
ds_test = ds_test.map(preprocessing).batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
```

### **Step 3: Build Model**
BigDL-Nano's `Model` (`bigdl.nano.tf.keras.Model`) and `Sequential` (`bigdl.nano.tf.keras.Sequential`) classes have identical APIs with `tf.keras.Model` and `tf.keras.Sequential`.
Here we initialize the model with pre-trained ImageNet weights, and we fine-tune it on the Stanford Dogs dataset.
```python
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
data_augmentation = Sequential([
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ])
def make_model(learning_rate=1e-2):
    inputs = layers.Input(shape = (img_size, img_size, 3))

    x = data_augmentation(inputs)
    backbone = EfficientNetB0(include_top=False, input_tensor=x)

    backbone.trainable = False

    x = layers.GlobalAveragePooling2D(name='avg_pool')(backbone.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = Model(inputs, outputs, name='EfficientNet')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy']
    )
    return model

def unfreeze_model(model):
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy']
    )
```

### **Step 4: Training**
```python
steps_per_epoch = ds_info.splits['train'].num_examples // batch_size
model_default = make_model()

model_default.fit(ds_train,
                  epochs=15,
                  validation_data=ds_test,
                  steps_per_epoch=steps_per_epoch)
unfreeze_model(model_default)
his_default = model_default.fit(ds_train,
                                epochs=10,
                                validation_data=ds_test,
                                steps_per_epoch=steps_per_epoch)
```
#### Multi-Instance Training
BigDL-Nano makes it very easy to conduct multi-instance training correctly. You can just set the `num_processes` parameter in the `fit` method in your `Model` or `Sequential` object and BigDL-Nano will launch the specific number of processes to perform data-parallel training.
```python
model_multi = make_model()

model_multi.fit(ds_train,
                epochs=15, 
                validation_data=ds_test, 
                steps_per_epoch=steps_per_epoch,
                num_processes=4, 
                backend='multiprocessing')
unfreeze_model(model_multi)
his_multi = model_multi.fit(ds_train,
                epochs=10,
                validation_data=ds_test, 
                steps_per_epoch=steps_per_epoch,
                num_processes=4, 
                backend='multiprocessing')
```

You can find the detailed result of training from [here](https://github.com/intel-analytics/BigDL/blob/main/python/nano/notebooks/tensorflow/tutorial/tensorflow_fit.ipynb)