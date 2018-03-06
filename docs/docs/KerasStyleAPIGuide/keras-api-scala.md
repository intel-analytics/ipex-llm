---
## **Introduction**
We hereby introduce a new set of __Keras-Style API__ based on [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) in BigDL for the sake of user-friendliness. Users, especially those familiar with Keras, are recommended to use the new API to create a BigDL model and train, evaluate or tune it in a distributed fashion.

To define a model in Scala, now one just need to import the package `bigdl.nn.keras.layer`.

One of the highlighted features with regard to the new API is __shape inference__. Users only need to specify the input shape (a `Shape` object __excluding__ batch dimension, for example, `inputShape=Shape(3, 4)` for 3D input) for the first layer of a model and for the remaining layers, the input dimension will be automatically inferred.

---
## **Sequential API**
The model is described as a linear stack of layers in the sequential API. Layers can be added into the `Sequential` container one by one and the order of the layers in the model will be the same as the insertion order.

To create a sequential container:
```scala
Sequential()
```

Example code to create a sequential model:
```scala
import com.intel.analytics.bigdl.nn.keras.{Dense, Sequential}
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
Input(name = null, inputShape = null)
```
Parameters:

* `name`: String to specify the name of the input node. Default is null.
* `shape`: A `Shape` object indicating the shape of the input node, not including batch.

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

val input = Input[Float](inputShape = Shape(128))
val dense = Dense[Float](32, activation = "relu").inputs(input)
val output = Dense[Float](10).inputs(dense)
val model = Model[Float](input, output)
```


---
## **Layers**
See [here](Layers/core.md) for all the available layers for the new set of Keras-Style API.

