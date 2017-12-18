# **Keras Support**

For __Python__ users, BigDL supports loading pre-defined Keras models. After loading a model, you can train, evaluate or tune a model on BigDL in a distributed manner.

The Keras version we support and test is [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) with TensorFlow backend.

You may refer to Python User Guide on how to [install](../PythonUserGuide/install-from-pip.md) and [run](../PythonUserGuide/run-from-pip.md) BigDL for Python users.

## **Load a Keras model into BigDL**

A Keras model definition in __JSON__ file can be loaded as a BigDL model.
Saved weights in __HDF5__ file can also be loaded together with the architecture of a model.
See [here](https://faroit.github.io/keras-docs/1.2.2/getting-started/faq/#how-can-i-save-a-keras-model) on how to save the architecture and weights of a Keras model.

You can directly call the API `load_keras` to load the Keras model into BigDL.

__Remark__: `keras==1.2.2` is required. If you need to load weights, you also need to install `h5py`. These packages can be installed via `pip` easily.

```python
from bigdl.nn.layer import *

bigdl_model = Model.load_keras(json_path=None, hdf5_path=None, by_name=False)
```
Parameters:

* `json_path` The JSON file path containing the Keras model definition to be loaded. Default to be `None` if you choose to load the Keras model from HDF5 file.
* `hdf5_path` The HDF5 file path containing the pre-trained Keras model weights with or without the model architecture. Default to be `None` if you choose to only load the model definition from JSON but not to load weights. In this case, initialized weights will be used for the model.
* `by_name`  Whether to load the weights of layers by name. Use this option only when you provide a HDF5 file. Default to be `False`, meaning that  weights are loaded based on the network's execution order topology. Otherwise, if it is set to be `True`, only those layers with the same name will be loaded with weights.

__NOTES__:

Please provide either `json_path` or `hdf5_path` when you call `load_keras`. You can provide `json_path` only to just load the model definition. You can provide `json_path` and `hdf5_path` together if you have separate files for the model architecture and pre-trained weights. Also, you can provide `hdf5_path` only if you save the model architecture and weights in a single HDF5 file.

JSON and HDF5 files can be loaded from any Hadoop-supported file system URI. For example,
```python
# load from local file system
bigdl_model = Model.load_keras(json_path="/tmp/model.json")
# load from HDFS
bigdl_model = Model.load_keras(hdf5_path="hdfs://...")
# load from S3
bigdl_model = Model.load_keras(hdf5_path="s3://...")
```

## **LeNet Example**

Here we show a simple example on how to load a Keras model into BigDL. The model used in this example is a [CNN](https://github.com/fchollet/keras/blob/1.2.2/examples/mnist_cnn.py) from Keras 1.2.2.

```python
# Define a CNN model in Keras 1.2.2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

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

# Save the Keras model definition to JSON
model_json = keras_model.to_json()
def_path = "/tmp/lenet.json"
with open(def_path, "w") as json_file:
    json_file.write(model_json)

# Load the JSON file to a BigDL model
from bigdl.nn.layer import *
bigdl_model = Model.load_keras(def_path=def_path)
```
After loading the model into BigDL, you can train it with MNIST dataset. See [here](../../../pyspark/bigdl/examples/keras/mnist_cnn.py) for the full example code which includes the training and validation after model loading. After 12 epochs, accuracy >97% can be achieved.

You can find several more examples [here](../../../pyspark/bigdl/examples/keras/) to get familiar with loading a Keras model into BigDL.

### **Limitations**
We have tested the model loading functionality with several standard [Keras applications](https://faroit.github.io/keras-docs/1.2.2/applications/) and [examples](https://github.com/fchollet/keras/tree/1.2.2/examples).

However, there still exist some arguments for Keras layers that are not supported in BigDL for now. Also we haven't supported self-defined Keras layers, but one can still define your customized layer converter and weight converter method for new layers if you wish.

In our future work, we will continue to add functionality and better support running Keras on BigDL.