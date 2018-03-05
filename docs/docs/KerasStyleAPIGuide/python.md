## __Introduction__ ##

We hereby introduce a new set of __Keras-Style API__ based on [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) in BigDL for the sake of user-friendliness. Users, especially those familiar with Keras, are strongly recommended to use the new API to create a BigDL model and train, evaluate or tune it in a distributed fashion.

To define a model in Python, now you just need to import the package `bigdl.nn.keras.layer`.

One of the highlighted features of the new API is __shape inference__. Users only need to specify the input shape (a shape tuple __excluding__ batch dimension, for example, `input_shape=(3, 4)` for 3D input) for the first layer of a model and for the remaining layers, the input dimension will be automatically inferred.
<br>

## __Sequential API__ ##
The model is described as a linear stack of layers in the sequential API. You can add the layers into the Sequential container one by one and the order of the layers in the model will be the same as the insertion order.
```python
from bigdl.nn.keras.layer import Sequential, Dense, Activation

model = Sequential()
model.add(Dense(32, input_shape=(128, )))
model.add(Activation("relu"))
model.get_output_shape() # (None, 32)
```

## __Functional API__ ##
The model is described as a graph in the functional API. It is more convenient than the sequential API when you want to define some complex model (for example, a model with multiple outputs).
```python
from bigdl.nn.keras.layer import Model, Input, Dense, merge

# create input nodes
input1 = Input(shape=(8, )) 
input2 = Input(shape=(6, ))
# pass an input node into a layer and get an output node
dense1 = Dense(10)(input1)
dense2 = Dense(10)(input2)
# merge two nodes following some merge mode
output = merge([dense1, dense2], mode="sum")
model = Model([input1, input2], output)
model.get_output_shape() # (None, 10)
```

## __Layers__ ##
See [here](Layers/core.md) for all the available layers for the new set of Keras-Style API.
<br>

## __Keras Code Support__ ##
If you have an existing piece of Keras code for a model definition, without installing Keras, you can directly migrate the code to construct a BigDL model by just replacing Keras import lines with:

`from bigdl.nn.keras.layer import *`, subject to the following limitations:

1. The Keras version we support and test is [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) with TensorFlow backend.

2. There exist some arguments supported in Keras layers but not supported in BigDL for now. See [here](../../APIGuide/keras-issues/#unsupported-layer-arguments) for the full list of unsupported layer arguments. Also, currently we haven't supported self-defined Keras layers or or [`Lambda`](https://faroit.github.io/keras-docs/1.2.2/layers/core/#lambda) layers.

3. The default dim_ordering in BigDL is `th` (Channel First, channel_axis=1).

4. Keras [backend](https://faroit.github.io/keras-docs/1.2.2/backend/) related code needs to be deleted or refactored appropriately.

5. Code involving Keras utility functions or loading weights from HDF5 files should be removed.

__Remark:__ We have tested for migrating Keras code definition of [VGG16](https://faroit.github.io/keras-docs/1.2.2/applications/#vgg16), [VGG19](https://faroit.github.io/keras-docs/1.2.2/applications/#vgg19), [ResNet50](https://faroit.github.io/keras-docs/1.2.2/applications/#resnet50) and [InceptionV3](https://faroit.github.io/keras-docs/1.2.2/applications/#inceptionv3) into BigDL.
<br>

## __Methods__ ##
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

```python
merge(inputs, mode="sum", concat_axis=-1, name=None)
```
Functional merge. Only use this method if you are defining a model with [Functional API](#functional-api).

Used to merge a list of input __nodes__ into a single output __node__ (__NOT__ layers!), following some merge mode.

Parameters:

* `inputs`: A list of node instances. Must be more than one node.
* `mode`: Merge mode. String, must be one of: 'sum', 'mul', 'concat', 'ave', 'cos','dot', 'max'. Default is 'sum'.
* `concat_axis`: Int, axis to use when concatenating nodes. Only specify this when the mode is 'concat'. Default is -1, meaning the last axis of the input.
* `name`: String to specify the name of merge. Default is None.