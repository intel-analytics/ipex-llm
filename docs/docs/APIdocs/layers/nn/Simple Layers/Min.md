## Min ##

**Scala:**
```scala
val min = Min(dim)
```
**Python:**
```python
min = Min(dim)
```

Applies a min operation over dimension `dim`.

**Parameters:**
* **dim** - A integer. The dimension to min along.
* **numInputDims** - An optional integer indicating the number of input dimensions.
 

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
val min = Min[Float](2)
val input = Tensor[Float](T(
 T(1.0f, 2.0f),
 T(3.0f, 4.0f))
)
val target = Tensor[Float](T(
 1.0f,
 1.0f
))
val output = min.forward(input)
val gradient = min.backward(input, target)
-> print(output)
1.0
3.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2]

-> print(gradient)
1.0     0.0     
1.0     0.0     
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
min = Min(2)
input = np.array([
  [1.0, 2.0],
  [3.0, 4.0]
])

target = np.array([1.0, 1.0])
output = min.forward(input)
gradient = min.backward(input, target)
-> print output
[ 1.  3.]
-> print gradient
[[ 1.  0.]
 [ 1.  0.]]
```