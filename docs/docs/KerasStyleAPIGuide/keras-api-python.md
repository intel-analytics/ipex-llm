## **Introduction**
We provide __Keras-Style API__ based on [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) in Analytics Zoo for the sake of user-friendliness. Users, especially those familiar with Keras, can easily use our API to create an Analytics Zoo model and train, evaluate or tune it in a distributed fashion.

To define a model in Python using the Keras-Style API, now one just need to import the following packages:

```python
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *
```

One of the highlighted features with regard to the Keras-Style API is __shape inference__. Users only need to specify the input shape (a shape tuple __excluding__ batch dimension, for example, `input_shape=(3, 4)` for 3D input) for the first layer of a model and for the remaining layers, the input dimension will be automatically inferred.

---
## **Define a model**
You can define a model either using [Sequential API](#sequential-api) or [Functional API](#functional-api). Remember to specify the input shape for the first layer.

After creating a model, you can call the following __methods__:

```python
get_input_shape()
```
```python
get_output_shape()
```
* Return the input or output shape of a model, which is a shape tuple. The first entry is `None` representing the batch dimension. For a model with multiple inputs or outputs, a list of shape tuples will be returned.

```python
set_name(name)
```
* Set the name of the model. Can alternatively specify the argument `name` in the constructor when creating a model.

---
## **Sequential API**
The model is described as a linear stack of layers in the Sequential API. Layers can be added into the `Sequential` container one by one and the order of the layers in the model will be the same as the insertion order.

To create a sequential container:
```python
Sequential()
```

Example code to create a sequential model:
```python
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(32, input_shape=(128, )))
model.add(Activation("relu"))
```

---
## **Functional API**
The model is described as a graph in the Functional API. It is more convenient than the Sequential API when defining some complex model (for example, a model with multiple outputs).

To create an input node:
```python
Input(shape=None, name=None)
```
Parameters:

* `shape`: A shape tuple indicating the shape of the input node, not including batch.
* `name`: String to set the name of the input node. If not specified, its name will by default to be a generated string.

To create a graph container:
```python
Model(input, output)
```
Parameters:

* `input`: An input node or a list of input nodes.
* `output`: An output node or a list of output nodes.

To merge a list of input __nodes__ (__NOT__ layers), following some merge mode in the Functional API:
```python
merge(inputs, mode="sum", concat_axis=-1) # This will return an output NODE.
```

Parameters:

* `inputs`: A list of node instances. Must be more than one node.
* `mode`: Merge mode. String, must be one of: 'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'. Default is 'sum'.
* `concat_axis`: Int, axis to use when concatenating nodes. Only specify this when merge mode is 'concat'. Default is -1, meaning the last axis of the input.

Example code to create a graph model:
```python
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras.layers import Input, Dense, merge

# instantiate input nodes
input1 = Input(shape=(8, )) 
input2 = Input(shape=(6, ))
# pass an input node into a layer and get an output node
dense1 = Dense(10)(input1)
dense2 = Dense(10)(input2)
# merge two nodes following some merge mode
output = merge([dense1, dense2], mode="sum")
# create a graph container
model = Model([input1, input2], output)
```

---
## **Layers**
See [here](Layers/core.md) for all the available layers for the Keras-Style API.

To set the name of a layer, you can either call `set_name(name)` or alternatively specify the argument `name` in the constructor when creating a layer.

---
## **LeNet Example**
Here we adopt our Keras-Style API to define a LeNet CNN model to be trained on the MNIST dataset:

```python
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import *

model = Sequential()
model.add(Reshape((1, 28, 28), input_shape=(28, 28, 1)))
model.add(Convolution2D(6, 5, 5, activation="tanh", name="conv1_5x5"))
model.add(MaxPooling2D())
model.add(Convolution2D(12, 5, 5, activation="tanh", name="conv2_5x5"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100, activation="tanh", name="fc1"))
model.add(Dense(10, activation="softmax", name="fc2"))

model.get_input_shape() # (None, 28, 28, 1)
model.get_output_shape() # (None, 10)
```

---
## **Keras Code Support**
If you have an existing piece of Keras code for a model definition, without installing Keras, you can directly migrate the code to construct an Analytics Zoo model by just replacing Keras import lines with:

```python
from zoo.pipeline.api.keras.models import *
from zoo.pipeline.api.keras.layers import *
```

and making modifications subject to the following limitations:

1. The Keras version we support and test is [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) with TensorFlow backend.

2. There exist some arguments supported in Keras layers but not supported in Analytics Zoo for now. See [here](https://bigdl-project.github.io/master/#APIGuide/keras-issues/#unsupported-layer-arguments) for the full list of unsupported layer arguments. 

3. The default dim_ordering in Analytics Zoo is `th` (Channel First, channel_axis=1).

4. Keras [backend](https://faroit.github.io/keras-docs/1.2.2/backend/) related code needs to be deleted or refactored appropriately.

5. Code involving Keras utility functions or loading weights from HDF5 files should be removed.

__Remark:__ We have tested for migrating Keras code definition of [VGG16](https://faroit.github.io/keras-docs/1.2.2/applications/#vgg16), [VGG19](https://faroit.github.io/keras-docs/1.2.2/applications/#vgg19), [ResNet50](https://faroit.github.io/keras-docs/1.2.2/applications/#resnet50) and [InceptionV3](https://faroit.github.io/keras-docs/1.2.2/applications/#inceptionv3) into Analytics Zoo.
