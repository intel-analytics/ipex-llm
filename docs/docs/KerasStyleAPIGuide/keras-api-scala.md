## **Introduction**
We hereby introduce a new set of __Keras-Style API__ based on [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) in BigDL for the sake of user-friendliness. Users, especially those familiar with Keras, are recommended to use the new API to create a BigDL model and train, evaluate or tune it in a distributed fashion.

To define a model in Scala using the Keras-Style API, now one just need to import the following packages:

```scala
import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.utils.Shape
```

One of the highlighted features with regard to the new API is __shape inference__. Users only need to specify the input shape (a [`Shape`](#shape) object __excluding__ batch dimension, for example, `inputShape=Shape(3, 4)` for 3D input) for the first layer of a model and for the remaining layers, the input dimension will be automatically inferred.

---
## **Shape**
Input and output shapes of a model in the Keras-Style API are described by the `Shape` object in Scala, which can be classified into `SingleShape` and `MultiShape`.

`SingleShape` is just a list of Int indicating shape dimensions while `MultiShape` is essentially a list of `Shape`.

Example code to create a shape:
```scala
// create a SingleShape
val shape1 = Shape(3, 4)
// create a MultiShape consisting of two SingleShape
val shape2 = Shape(List(Shape(1, 2, 3), Shape(4, 5, 6)))
```
You can use method `toSingle()` to cast a `Shape` to a `SingleShape`. Similarly, use `toMulti()` to cast a `Shape` to a `MultiShape`.

---
## **Define a model**
You can define a model either using [Sequential API](#sequential-api) or [Functional API](#functional-api). Remember to specify the input shape for the first layer.

After creating a model, you can call the following __methods__:

```scala
getInputShape()
```
```scala
getOutputShape()
```
* Return the input or output shape of a model, which is a [`Shape`](#shape) object. For `SingleShape`, the first entry is `-1` representing the batch dimension. For a model with multiple inputs or outputs, it will return a `MultiShape`.

```scala
setName(name)
```
* Set the name of the model.

See [here](Optimization/training/) on how to train, predict or evaluate a defined model.

---
## **Sequential API**
The model is described as a linear stack of layers in the Sequential API. Layers can be added into the `Sequential` container one by one and the order of the layers in the model will be the same as the insertion order.

To create a sequential container:
```scala
Sequential()
```

Example code to create a sequential model:
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Dense, Activation}
import com.intel.analytics.bigdl.utils.Shape

val model = Sequential[Float]()
model.add(Dense(32, inputShape = Shape(128)))
model.add(Activation("relu"))
```

---
## **Functional API**
The model is described as a graph in the Functional API. It is more convenient than the Sequential API when defining some complex model (for example, a model with multiple outputs).

To create an input node:
```scala
Input(inputShape = null, name = null)
```
Parameters:

* `inputShape`: A [`Shape`](#shape) object indicating the shape of the input node, not including batch.
* `name`: String to set the name of the input node. If not specified, its name will by default to be a generated string.

To create a graph container:
```scala
Model(input, output)
```
Parameters:

* `input`: An input node or an array of input nodes.
* `output`: An output node or an array of output nodes.

To merge a list of input __nodes__ (__NOT__ layers), following some merge mode in the Functional API:
```scala
import com.intel.analytics.bigdl.nn.keras.Merge.merge

merge(inputs, mode = "sum", concatAxis = -1) // This will return an output NODE.
```

Parameters:

* `inputs`: A list of node instances. Must be more than one node.
* `mode`: Merge mode. String, must be one of: 'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'. Default is 'sum'.
* `concatAxis`: Int, axis to use when concatenating nodes. Only specify this when merge mode is 'concat'. Default is -1, meaning the last axis of the input.

Example code to create a graph model:
```scala
import com.intel.analytics.bigdl.nn.keras.{Input, Dense, Model}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.nn.keras.Merge.merge

// instantiate input nodes
val input1 = Input[Float](inputShape = Shape(8))
val input2 = Input[Float](inputShape = Shape(6))
// call inputs() with an input node and get an output node
val dense1 = Dense[Float](10).inputs(input1)
val dense2 = Dense[Float](10).inputs(input2)
// merge two nodes following some merge mode
val output = merge(inputs = List(dense1, dense2), mode = "sum")
// create a graph container
val model = Model[Float](Array(input1, input2), output)
```

---
## **Layers**
See [here](Layers/core.md) for all the available layers for the new set of Keras-Style API.

To set the name of a layer, call the method `setName(name)` of the layer.

---
## **LeNet Example**
Here we adopt our Keras-Style API to define a LeNet CNN model to be trained on the MNIST dataset:

```scala
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.utils.Shape

val model = Sequential()
model.add(Reshape(Array(1, 28, 28), inputShape = Shape(28, 28, 1)))
model.add(Convolution2D(6, 5, 5, activation = "tanh").setName("conv1_5x5"))
model.add(MaxPooling2D())
model.add(Convolution2D(12, 5, 5, activation = "tanh").setName("conv2_5x5"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100, activation = "tanh").setName("fc1"))
model.add(Dense(10, activation = "softmax").setName("fc2"))

model.getInputShape().toSingle().toArray // Array(-1, 28, 28, 1)
model.getOutputShape().toSingle().toArray // Array(-1, 10)
```
See [here](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/keras) for detailed introduction of LeNet, the full example code and running instructions.