---
## **Highway**
Densely connected highway network.

Highway layers are a natural extension of LSTMs to feedforward networks.

The input of this layer should be 2D, i.e. (batch, input dim).

**Scala:**
```scala
Highway(activation = null, wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
Highway(activation=None, W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

Parameters:

* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `wRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.Highway
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Highway(inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.26041138	0.4286919	1.723103
1.4516269	0.5557163	-0.1149741
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
-0.006746907	-0.109112576	1.3375516
0.6065166	0.41575465	-0.06849813
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Highway
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Highway(input_shape = (3)))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.5762107  0.45679288 0.00370956]
 [0.24133312 0.38104653 0.05249192]]
```
Output is:
```python
[[0.5762107  0.4567929  0.00370956]
 [0.24133313 0.38104653 0.05249191]]
```
