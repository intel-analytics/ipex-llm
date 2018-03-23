## **Activation**
Simple activation function to be applied to the output.

**Scala:**
```scala
Activation(activation, inputShape = null)
```
**Python:**
```python
Activation(activation, input_shape=None, name=None)
```

Parameters:

* `activation`: Name of the activation function as string. See [here](#available-activations) for available activation strings.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Activation}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Activation("tanh", inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
2.1659365	0.28006053	-0.20148286
0.9146865	 3.4301455	  1.0930616
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.9740552	 0.2729611	  -0.1988
 0.723374	0.99790496	0.7979928
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Activation

model = Sequential()
model.add(Activation("tanh", input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[ 0.26202468  0.15868397  0.27812652]
 [ 0.45931689  0.32100054  0.51839282]]
```
Output is
```python
[[ 0.2561883   0.15736534  0.27117023]
 [ 0.42952728  0.31041133  0.47645861]]
```

Note that the following two pieces of code will be equivalent:
```python
model.add(Dense(32))
model.add(Activation('relu'))
```
```python
model.add(Dense(32, activation="relu"))
```


---
## **Available Activations**
* [relu](../../APIGuide/Layers/Activations/#relu)
* [tanh](../../APIGuide/Layers/Activations/#tanh)
* [sigmoid](../../APIGuide/Layers/Activations/#sigmoid)
* [hard_sigmoid](../../APIGuide/Layers/Activations/#hardsigmoid)
* [softmax](../../APIGuide/Layers/Activations/#softmax)
* [softplus](../../APIGuide/Layers/Activations/#softplus)
* [softsign](../../APIGuide/Layers/Activations/#softsign)