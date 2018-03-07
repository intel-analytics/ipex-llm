---
## **InputLayer**
Can be used as an entry point into a model.

**Scala:**
```scala
InputLayer(inputShape = null, name = null)
```
**Python:**
```python
InputLayer(input_shape=None, name=None)
```
Parameters:

* `inputShape`: For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.
* `name`: String to specify the name of the input layer. Default is null.


---
## **Dense**
A densely-connected NN layer.

The most common input is 2D.

**Scala:**
```scala
Dense(outputDim, init = "glorot_uniform", activation = null, wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
Dense(output_dim, init="glorot_uniform", activation=None, W_regularizer=None, b_regularizer=None, bias=True, input_dim=None, input_shape=None, name=None)
```

**Parameters:**

* `outputDim`: The size of the output dimension.
* `init`: String representation of the initialization method for the weights of the layer. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is "glorot_uniform".
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `wRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.
* `name`: This parameter is for Python API only. String to specify the name of the layer. Default is None. For Scala API, please use `setName(name)` instead.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Dense}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(Dense(5, activation = "relu", inputShape = Shape(4)))
val input = Tensor[Float](2, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
 1.8646977	-0.059090078  0.091468036   0.6387431
-0.4485392	   1.5150243  -0.60176533  -0.6811443
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
1.0648216	      0.0  0.0  0.0         0.0
      0.0  0.20690927  0.0  0.0  0.34191078
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x5]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.layer import Sequential, Dense

model = Sequential()
model.add(Dense(5, activation="relu", input_shape=(4, )))
input = np.random.random([2, 4])
output = model.forward(input)
```
Input is:
```python
[[ 0.26202468  0.15868397  0.27812652  0.45931689]
 [ 0.32100054  0.51839282  0.26194293  0.97608528]]
```
Output is
```python
[[ 0.  0.  0.  0.02094215  0.38839486]
 [ 0.  0.  0.  0.24498197  0.38024583]]
```