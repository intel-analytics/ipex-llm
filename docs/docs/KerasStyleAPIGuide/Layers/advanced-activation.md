## **ELU**
Exponential Linear Unit.

It follows: f(x) =  alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0.

**Scala:**
```scala
ELU(alpha = 1.0, inputShape = null)
```
**Python:**
```python
ELU(alpha=1.0, input_shape=None, name=None)
```

**Parameters:**

* `alpha`: Scale for the negative factor. Default is 1.0.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.ELU
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(ELU[Float](inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.13405465	0.05160992	-1.4711418
1.5808829	-1.3145303	0.6709266
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
-0.1254577	0.05160992	-0.77033687
1.5808829	-0.73139954	0.6709266
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import ELU
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(ELU(input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.90404922 0.23530925 0.49711093]
 [0.43009161 0.22446032 0.90144771]]
```
Output is
```python
[[0.9040492  0.23530924 0.49711093]
 [0.43009162 0.22446032 0.9014477 ]]
```
