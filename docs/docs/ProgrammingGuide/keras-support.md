# **Keras Support**

BigDL supports loading pre-defined Keras models and running the models in a distributed manner.

The Keras version we support and test is [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/).

## **Load a Keras model into BigDL**

A Keras model definition in __JSON__ file can be loaded as a BigDL model.
Saved weights in __HDF5__ file can also be loaded together with the architecture of a model.
See [here](https://faroit.github.io/keras-docs/1.2.2/getting-started/faq/#how-can-i-save-a-keras-model) on how to save the architecture and weights of a Keras model.

You can directly use the API `load_keras` to load the Keras model into BigDL.

__Remark__: Packages `tensorflow`, `keras==1.2.2` and `h5py` are required. They can be installed via `pip` easily.

```python
from bigdl.nn.layer import *

bigdl_model = Model.load_keras(def_path, weights_path=None, by_name=False)
```
Parameters:

* `def_path` The JSON file path containing the keras model definition to be loaded.
* `weights_path`  The HDF5 file path containing the pre-trained keras model weights. Default to be `None` if you choose not to load weights. In this case, initialized weights will be used for the model.
* `by_name`  Whether to load the weights of layers by name. Use this option only when you load the pre-trained weights. Default to be `False`, meaning that  weights are loaded based on the network's execution order topology. Otherwise, if it is set to be `True`, only those layers with the same name will be loaded with weights.

## **Example**

Here we illustrate with a concrete [LeNet example](https://github.com/fchollet/keras/blob/1.2.2/examples/mnist_cnn.py) from Keras.

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

# Define a convnet model in Keras
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
def_path = "lenet.json"
with open(def_path, "w") as json_file:
    json_file.write(model_json)

from bigdl.util.common import *
from bigdl.nn.layer import *
from bigdl.optim.optimizer import *
from bigdl.nn.criterion import *

# Load the JSON file to a BigDL model
bigdl_model = Model.load_keras(def_path=def_path)

# Load data from Keras MNIST dataset
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train /= 255
X_test /= 255

# Distribute data to form RDDs
sc = get_spark_context(conf=create_spark_conf())
redire_spark_logs()
show_bigdl_info_logs()
init_engine()
X_train = sc.parallelize(X_train)
y_train = sc.parallelize(y_train)
train_data = X_train.zip(y_train).map(lambda t: Sample.from_ndarray(t[0], t[1] + 1))
X_test = sc.parallelize(X_test)
y_test = sc.parallelize(y_test)
test_data = X_test.zip(y_test).map(lambda t: Sample.from_ndarray(t[0], t[1] + 1))
optimizer = Optimizer(
    model=bigdl_model,
    training_rdd=train_data,
    criterion=ClassNLLCriterion(),
    optim_method=SGD(learningrate=0.01),
    end_trigger=MaxEpoch(12),
    batch_size=128)
optimizer.set_validation(
    batch_size=128,
    val_rdd=test_data,
    trigger=EveryEpoch(),
    val_method=[Top1Accuracy()]
)
optimizer.optimize()
```

### **Limitations**
We have tested the model loading functionality with some standard [Keras applications](https://faroit.github.io/keras-docs/1.2.2/applications/) and [examples](https://github.com/fchollet/keras/tree/1.2.2/examples).

There still exist some arguments for Keras layers that are not supported in BigDL for now. We haven't supported self-defined Keras layers, but one can still define your customized layer converter and weight converter method for new layers if you wish.

In our future work, we will continue add functionality and better support running Keras on BigDL.