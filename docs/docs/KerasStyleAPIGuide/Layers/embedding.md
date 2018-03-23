## **Embedding**
Turn positive integers (indexes) into dense vectors of fixed size.

The input of this layer should be 2D.

**Scala:**
```scala
Embedding(inputDim, outputDim, init = "uniform", wRegularizer = null, inputShape = null)
```
**Python:**
```python
Embedding(input_dim, output_dim, init="uniform", W_regularizer=None, input_shape=None, name=None)
```

**Parameters:**

* `inputDim`: Int > 0. Size of the vocabulary.
* `outputDim`: Int >= 0. Dimension of the dense embedding.
* `init`: String representation of the initialization method for the weights of the layer. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is "uniform".
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Embedding}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Embedding(8, 2, inputShape = Shape(4)))
val input = Tensor[Float](2, 4)
input(Array(1, 1)) = 1
input(Array(1, 2)) = 2
input(Array(1, 3)) = 4
input(Array(1, 4)) = 5
input(Array(2, 1)) = 4
input(Array(2, 2)) = 3
input(Array(2, 3)) = 2
input(Array(2, 4)) = 6
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0	2.0	4.0	5.0
4.0	3.0	2.0	6.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.03256504	    -0.043232664
-0.044753443	0.026075097
0.045668535	    0.02456015
0.021222712	    -0.04373116

(2,.,.) =
0.045668535	    0.02456015
0.03761902	    -0.0014174521
-0.044753443	0.026075097
-0.030343587	-0.0015718295

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x2]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Embedding

model = Sequential()
model.add(Embedding(8, 2, input_shape=(4,)))
input = np.random.randint(4, size=(2, 4))
output = model.forward(input)
```
Input is:
```python
[[0 2 2 2]
 [2 1 1 0]]
```
Output is
```python
[[[ 0.0094721  -0.01927968]
  [-0.00483634 -0.03992473]
  [-0.00483634 -0.03992473]
  [-0.00483634 -0.03992473]]

 [[-0.00483634 -0.03992473]
  [-0.03603687 -0.03708585]
  [-0.03603687 -0.03708585]
  [ 0.0094721  -0.01927968]]]
```
