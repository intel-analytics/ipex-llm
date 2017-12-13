# **Keras Support**

For __Python__ users, BigDL supports loading pre-defined Keras models. After loading a model, you can train, evaluate or tune a model on BigDL in a distributed manner.

The Keras version we support and test is [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) with TensorFlow backend.

You may refer to Python User Guide on how to [install](../PythonUserGuide/install-from-pip.md) and [run](../PythonUserGuide/run-from-pip.md) BigDL for Python users.

## **Load a Keras model into BigDL**

A Keras model definition in __JSON__ file can be loaded as a BigDL model.
Saved weights in __HDF5__ file can also be loaded together with the architecture of a model.
See [here](https://faroit.github.io/keras-docs/1.2.2/getting-started/faq/#how-can-i-save-a-keras-model) on how to save the architecture and weights of a Keras model.

You can directly use the API `load_keras` to load the Keras model into BigDL.

__Remark__: Packages `keras==1.2.2`, `tensorflow` and `h5py` are required. They can be installed via `pip` easily.

```python
from bigdl.nn.layer import *

bigdl_model = Model.load_keras(def_path, weights_path=None, by_name=False)
```
Parameters:

* `def_path` The JSON file path containing the keras model definition to be loaded.
* `weights_path`  The HDF5 file path containing the pre-trained keras model weights. Default to be `None` if you choose not to load weights. In this case, initialized weights will be used for the model.
* `by_name`  Whether to load the weights of layers by name. Use this option only when you load the pre-trained weights. Default to be `False`, meaning that  weights are loaded based on the network's execution order topology. Otherwise, if it is set to be `True`, only those layers with the same name will be loaded with weights.

We support loading model and weight files from any Hadoop-supported file system URI.
```python
# load from local file system
bigdl_model = Model.load_keras("/tmp/model.json")
# load from HDFS
bigdl_model = Model.load_keras("hdfs://...")
# load from S3
bigdl_model = Model.load_keras("s3://...")
```

## **LeNet Example**

Here we show a simple example on how to load a Keras model into BigDL. The model used in this example is a [CNN](https://github.com/fchollet/keras/blob/1.2.2/examples/mnist_cnn.py) from Keras.

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

# Define a CNN model in Keras
keras_model = Sequential()
keras_model.add(Convolution2D(32, 3, 3, border_mode='valid',
                              input_shape=(1, 28, 28)))
keras_model.add(Activation('relu'))
keras_model.add(Convolution2D(32, 3, 3))
keras_model.add(Activation('relu'))
keras_model.add(MaxPooling2D(pool_size=(2, 2)))
keras_model.add(Dropout(0.25))

keras_model.add(Flatten())
keras_model.add(Dense(128))
keras_model.add(Activation('relu'))
keras_model.add(Dropout(0.5))
keras_model.add(Dense(10))
keras_model.add(Activation('softmax'))

# Save the keras model to JSON file
model_json = keras_model.to_json()
def_path = "/tmp/lenet.json"
with open(def_path, "w") as json_file:
    json_file.write(model_json)

# Load the JSON file to a BigDL model
from bigdl.nn.layer import *
bigdl_model = Model.load_keras(def_path=def_path)
```
After loading the model into BigDL, you can train it with MNIST dataset. See [here](https://github.com/intel-analytics/BigDL/blob/master/pyspark/bigdl/keras/examples/keras_lenet.py) for the full example code which includes the training and validation after model load. After 12 epochs, accuracy >97% can be achieved.

### **Limitations**
We have tested the model loading functionality with several standard [Keras applications](https://faroit.github.io/keras-docs/1.2.2/applications/) and [examples](https://github.com/fchollet/keras/tree/1.2.2/examples).

However, there still exist some arguments for Keras layers that are not supported in BigDL for now. Also we haven't supported self-defined Keras layers, but one can still define your customized layer converter and weight converter method for new layers if you wish.

In our future work, we will continue to add functionality and better support running Keras on BigDL.