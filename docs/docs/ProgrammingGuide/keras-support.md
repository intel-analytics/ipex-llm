For __Python__ users, BigDL supports loading pre-defined Keras models.

The Keras version we support and test is [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) with TensorFlow backend. Up to now, we have generally supported __ALL__ its layers.

After loading a model into BigDL, you can train, evaluate or tune this model in a distributed manner. We have generally supported __ALL__ the [losses](https://faroit.github.io/keras-docs/1.2.2/objectives/) in Keras 1.2.2. See [here](../APIGuide/Losses.md) to find the corresponding criterions in BigDL.

If you haven't been familiar with BigDL yet, you may refer to Python User Guide on how to [install](../PythonUserGuide/install-from-pip.md) and [run](../PythonUserGuide/run-from-pip.md) BigDL for Python users before you start this page.

## **Load a Keras model into BigDL**

A Keras model definition in __JSON__ file can be loaded as a BigDL model.
Saved weights in __HDF5__ file can also be loaded together with the architecture of a Keras model.
See [here](https://faroit.github.io/keras-docs/1.2.2/getting-started/faq/#how-can-i-save-a-keras-model) on how to save a model in Keras.

You can directly call the API `Model.load_keras` to load a Keras model into BigDL.

__Remark__: `keras==1.2.2` is required beforehand. If you are to load a HDF5 file, you also need to install `h5py`. These packages can be installed via `pip` easily.

```python
from bigdl.nn.layer import *

bigdl_model = Model.load_keras(json_path=None, hdf5_path=None, by_name=False)
```
__Parameters__:

* `json_path` The JSON file path containing the Keras model definition to be loaded. Default to be `None` if you choose to load a Keras model from a HDF5 file.
* `hdf5_path` The HDF5 file path containing the pre-trained weights with or without the model architecture. Please use weights from Keras 1.2.2 with __`tensorflow backend`__. Default to be `None` if you choose to only load the model definition from JSON but not to load weights. In this case, BigDL will use initialized weights for the model.
* `by_name`  Whether to load the weights of layers by name. Use this option only when you provide a HDF5 file. Default to be `False`, meaning that  weights are loaded based on the network's execution order topology. Otherwise, if it is set to be `True`, only those layers with the same name will be loaded with weights.

__NOTES__:

* Please provide either `json_path` or `hdf5_path` when you call `Model.load_keras`. You can provide `json_path` only to just load the model definition. You can provide `json_path` and `hdf5_path` together if you have separate files for the model architecture and its pre-trained weights. Also, if you save the model architecture and its weights in a single HDF5 file, you can provide `hdf5_path` only.

* JSON and HDF5 files can be loaded from any Hadoop-supported file system URI. For example,
```python
# load from local file system
bigdl_model = Model.load_keras(json_path="/tmp/model.json", hdf5_path="/tmp/weights.h5")
# load from HDFS
bigdl_model = Model.load_keras(hdf5_path="hdfs://model.h5")
# load from S3
bigdl_model = Model.load_keras(hdf5_path="s3://model.h5")
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
path = "/tmp/lenet.json"
with open(def_path, "w") as json_file:
    json_file.write(model_json)

# Load the JSON file to a BigDL model
from bigdl.nn.layer import *
bigdl_model = Model.load_keras(json_path=path)
```
After loading the model into BigDL, you can train it with the MNIST dataset. See [here](https://github.com/intel-analytics/BigDL/blob/master/pyspark/bigdl/examples/keras/mnist_cnn.py) for the full example code which includes the training and validation after model loading. After 12 epochs, accuracy >97% can be achieved.

You can find several more examples [here](https://github.com/intel-analytics/BigDL/tree/master/pyspark/bigdl/examples/keras) to get familiar with loading a Keras model into BigDL. We will add more examples to this directory in the future.

## **Limitations**
We have tested the model loading functionality with several standard Keras [applications](https://faroit.github.io/keras-docs/1.2.2/applications/) and [examples](https://github.com/fchollet/keras/tree/1.2.2/examples).

However, there exist some arguments for Keras layers that are not supported in BigDL for now. Also, we haven't supported self-defined Keras layers, but one can still define your customized layer converter and weight converter methods for new layers if you wish. See [here](../APIGuide/keras-issues.md) for the full list of unsupported layer arguments and some known issues we have found so far.

In our future work, we will continue to add functionality and better support running Keras on BigDL.