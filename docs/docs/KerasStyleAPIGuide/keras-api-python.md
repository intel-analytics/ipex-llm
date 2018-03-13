---
## **Introduction**
We hereby introduce a new set of __Keras-Style API__ based on [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) in BigDL for the sake of user-friendliness. Users, especially those familiar with Keras, are recommended to use the new API to create a BigDL model and train, evaluate or tune it in a distributed fashion.

To define a model in Python using the Keras-Style API, now one just need to import the package

`bigdl.nn.keras.layer`

One of the highlighted features with regard to the new API is __shape inference__. Users only need to specify the input shape (a shape tuple __excluding__ batch dimension, for example, `input_shape=(3, 4)` for 3D input) for the first layer of a model and for the remaining layers, the input dimension will be automatically inferred.


---
## **Sequential API**
The model is described as a linear stack of layers in the sequential API. Layers can be added into the `Sequential` container one by one and the order of the layers in the model will be the same as the insertion order.

To create a sequential container:
```python
Sequential(name=None)
```
Parameters:

* `name`: String to specify the name of the sequential model. Default is None.

Example code to create a sequential model:
```python
from bigdl.nn.keras.layer import Sequential, Dense, Activation

model = Sequential()
model.add(Dense(32, input_shape=(128, )))
model.add(Activation("relu"))
```


---
## **Functional API**
The model is described as a graph in the functional API. It is more convenient than the sequential API when defining some complex model (for example, a model with multiple outputs).

To create an input node:
```python
Input(shape=None, name=None)
```
Parameters:

* `shape`: A shape tuple indicating the shape of the input node, not including batch.
* `name`: String to specify the name of the input node. Default is None.

To create a graph container:
```python
Model(input, output, name=None)
```
Parameters:

* `input`: An input node or a list of input nodes.
* `output`: An output node or a list of output nodes.
* `name`: String to specify the name of the graph model. Default is None.


Example code to create a graph model:
```python
from bigdl.nn.keras.layer import Model, Input, Dense, merge

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
See [here](Layers/core.md) for all the available layers for the new set of Keras-Style API.


---
## **Methods**
__Methods for a [sequential](#sequential-api) or a [functional](#functional-api) model:__

```python
get_output_shape()
```
Return the output shape of a model, which is a shape tuple. The first entry is `None` representing the batch dimension.

For a model with multiple outputs, it will return a list of shape tuples.

```python
get_input_shape()
```
Return the input shape of a model, which is a shape tuple. The first entry is `None` representing the batch dimension.

For a model with multiple inputs, it will return a list of shape tuples.

__Methods for a [functional](#functional-api) model only:__

```python
merge(inputs, mode="sum", concat_axis=-1, name=None)
```
Used to merge a list of input __nodes__ (__NOT__ layers), following some merge mode.

Return an output __node__.

Parameters:

* `inputs`: A list of node instances. Must be more than one node.
* `mode`: Merge mode. String, must be one of: 'sum', 'mul', 'concat', 'ave', 'cos','dot', 'max'. Default is 'sum'.
* `concat_axis`: Int, axis to use when concatenating nodes. Only specify this when merge mode is 'concat'. Default is -1, meaning the last axis of the input.
* `name`: String to specify the name of merge. Default is None.

__Methods for either a [layer](Layers/core.md) or a [model](#sequential-api):__

```python
set_name(name)
```
Set the name of a module. Can alternatively specify the parameter `name` when creating a layer or a model.

Parameters:

* `name`: String to specify the name.


---
## **LeNet Example**
Here we adopt our Keras-Style API to define a LeNet CNN model to be trained on the MNIST dataset:

```python
from bigdl.nn.keras.layer import *

model = Sequential()
model.add(Reshape((1, 28, 28), input_shape=(28, 28, 1)))
model.add(Convolution2D(32, 3, 3, activation="relu"))
model.add(Convolution2D(32, 3, 3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.get_input_shape() # (None, 28, 28, 1)
model.get_output_shape() # (None, 10)
```
See [here](https://github.com/intel-analytics/BigDL/tree/master/pyspark/bigdl/examples/lenet) for the full example code and running instructions.


---
## **Keras Code Support**
If you have an existing piece of Keras code for a model definition, without installing Keras, you can directly migrate the code to construct a BigDL model by just replacing Keras import lines with:

`from bigdl.nn.keras.layer import *` and making modifications subject to the following limitations:

1. The Keras version we support and test is [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) with TensorFlow backend.

2. There exist some arguments supported in Keras layers but not supported in BigDL for now. See [here](../../APIGuide/keras-issues/#unsupported-layer-arguments) for the full list of unsupported layer arguments. Also, currently we haven't supported self-defined Keras layers or [`Lambda`](https://faroit.github.io/keras-docs/1.2.2/layers/core/#lambda) layers.

3. The default dim_ordering in BigDL is `th` (Channel First, channel_axis=1).

4. Keras [backend](https://faroit.github.io/keras-docs/1.2.2/backend/) related code needs to be deleted or refactored appropriately.

5. Code involving Keras utility functions or loading weights from HDF5 files should be removed.

__Remark:__ We have tested for migrating Keras code definition of [VGG16](https://faroit.github.io/keras-docs/1.2.2/applications/#vgg16), [VGG19](https://faroit.github.io/keras-docs/1.2.2/applications/#vgg19), [ResNet50](https://faroit.github.io/keras-docs/1.2.2/applications/#resnet50) and [InceptionV3](https://faroit.github.io/keras-docs/1.2.2/applications/#inceptionv3) into BigDL.
