---
## **Embedding**
Turn positive integers (indexes) into dense vectors of fixed size.

The input of this layer should be 2D.

**Scala:**
```scala
Embedding(inputDim, outputDim, init = RandomUniform, wRegularizer = null, inputShape = null)
```
**Python:**
```python
Embedding(input_dim, output_dim, init="uniform", W_regularizer=None, input_shape=None)
```

**Parameters:**

* `inputDim`: Int > 0. Size of the vocabulary.
* `outputDim`: Int >= 0. Dimension of the dense embedding.
* `init`: String representation of the initialization method for the weights of the layer. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is RandomUniform.
* `wRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Embedding}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(Embedding(8, 2, inputShape = Shape(4)))
val input = Tensor[Float](2, 4).randn()
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
1.2897599	-0.041361827	-1.3856494	-0.8376702
0.41095054	1.7178319	0.5430337	1.8327101
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.020649161	-0.008993048
0.018447913	0.032980155
0.02699975	0.005203543
0.02510968	0.015812807

(2,.,.) =
0.02699975	0.005203543
0.03393969	0.011750275
0.018447913	0.032980155
-0.004595372	0.025243163

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x2]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Embedding

model = Sequential()
model.add(Embedding(8, 4, input_shape=(4)))
input = np.random.random([2, 4])

output = model.forward(input)
```
Input is:
```python
[[[0.06092268 0.0508438  0.47256153 0.80004565]
  [0.48706905 0.65704781 0.04297214 0.42288264]
  [0.92286158 0.85394381 0.46423248 0.87896669]]

 [[0.216527   0.13880484 0.93482372 0.44812419]
  [0.95553331 0.27084259 0.58913626 0.01879454]
  [0.6656435  0.1985877  0.94133745 0.57504128]]]
```
Output is
```python
[[[ 0.7461933  -2.3189526  -1.454972   -0.7323345   1.5272427  -0.87963724  0.6278059  -0.23403725]]

 [[ 1.2397771  -0.9249111  -1.1432207  -0.92868984  0.53766745 -1.0271561  -0.9593589  -0.4768026 ]]]
```

---