## **Introduction**
We hereby introduce a new set of __Keras-Style API__ based on [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) in BigDL for the sake of user-friendliness. Users, especially those familiar with Keras, are recommended to use the new API to create a BigDL model and train, evaluate or tune it in a distributed fashion.

To define a model in Scala using the Keras-Style API, now one just need to import the package

`com.intel.analytics.bigdl.nn.keras`

One of the highlighted features with regard to the new API is __shape inference__. Users only need to specify the input shape (a [`Shape`](#shape) object __excluding__ batch dimension, for example, `inputShape=Shape(3, 4)` for 3D input) for the first layer of a model and for the remaining layers, the input dimension will be automatically inferred.

---
## **Sequential API**
The model is described as a linear stack of layers in the sequential API. Layers can be added into the `Sequential` container one by one and the order of the layers in the model will be the same as the insertion order.

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
The model is described as a graph in the functional API. It is more convenient than the sequential API when defining some complex model (for example, a model with multiple outputs).

To create an input node:
```scala
Input(inputShape = null, name = null)
```
Parameters:

* `inputShape`: A [`Shape`](#shape) object indicating the shape of the input node, not including batch.
* `name`: String to specify the name of the input node. Default is null.

To create a graph container:
```scala
Model(input, output)
```
Parameters:

* `input`: An input node or an array of input nodes.
* `output`: An output node or an array of output nodes.


Example code to create a graph model:
```scala
import com.intel.analytics.bigdl.nn.keras.{Input, Dense, Model}
import com.intel.analytics.bigdl.utils.Shape

// instantiate an input node
val input = Input[Float](inputShape = Shape(128))
// call inputs() with an input node and get an output node
val dense = Dense[Float](32, activation = "relu").inputs(input)
val output = Dense[Float](10).inputs(dense)
// create a graph container
val model = Model[Float](input, output)
```


---
## **Shape**
Input and output shapes of a model in the new API will be described by the `Shape` object in Scala, which can be classified into `SingleShape` and `MultiShape`.

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
## **Layers**
See [here](Layers/core.md) for all the available layers for the new set of Keras-Style API.


---
## **Methods**
#### __Methods for a [sequential](#sequential-api) or a [functional](#functional-api) model:__

```scala
getOutputShape()
```
Return the output shape of a model, which is a [`Shape`](#shape) object.

For `SingleShape`, the first entry is `-1` representing the batch dimension.

For a model with multiple outputs, it will return a `MultiShape`.

```scala
getInputShape()
```
Return the input shape of a model, which is a [`Shape`](#shape) object.

For `SingleShape`, the first entry is `-1` representing the batch dimension.

For a model with multiple inputs, it will return a `MultiShape`.

#### __Methods for either a [layer](Layers/core.md) or a [model](#sequential-api):__

```scala
setName(name)
```
Set the name of a module.

Parameters:

* `name`: String to specify the name.


---
## **LeNet Example**
Here we adopt our Keras-Style API to define a LeNet CNN model to be trained on the MNIST dataset:

```scala
import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.utils.Shape

val model = Sequential[Float]()
model.add(Reshape(Array(1, 28, 28), inputShape = Shape(28, 28, 1)))
model.add(Convolution2D(32, 3, 3, activation = "relu"))
model.add(Convolution2D(32, 3, 3, activation = "relu"))
model.add(MaxPooling2D(poolSize = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.getInputShape().toSingle().toArray // Array(-1, 28, 28, 1)
model.getOutputShape().toSingle().toArray // Array(-1, 10)
```
See [here](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/keras) for the full example code and running instructions.