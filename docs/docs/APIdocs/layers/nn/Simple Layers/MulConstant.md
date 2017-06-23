## MulConstant ##

**Scala:**
```scala
val layer = MulConstant(const)
```
**Python:**
```python
layer = MulConstant(const)
```

Multiplies input Tensor by a (non-learnable) scalar constant.
This module is sometimes useful for debugging purposes.

**Parameters:**
* **constant** - scalar constant
* **inplace** - Can optionally do its operation in-place without using extra state memory. Default: false

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
val input = Tensor[Float](T(
 T(1.0f, 2.0f),
 T(3.0f, 4.0f))
)
val target = Tensor[Float](T(
 T(1.0f, 1.0f),
 T(1.0f, 1.0f))
)
val scalar = 2.0
val module = MulConstant[Float](scalar)
val output = module.forward(input)
val gradient = module.backward(input, target)
-> print(output)
2.0     4.0     
6.0     8.0     
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
     
-> print(gradient)
2.0     2.0     
2.0     2.0     
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
input = np.array([
          [1.0, 2.0],
          [3.0, 4.0]
        ])
target = np.array([
           [1.0, 1.0],
           [1.0, 1.0]
         ])
scalar = 2.0
module = MulConstant(scalar)
output = module.forward(input)
gradient = module.backward(input, target)
-> print output
[[ 2.  4.]
 [ 6.  8.]]
-> print gradient
[[ 2.  2.]
 [ 2.  2.]]
```
