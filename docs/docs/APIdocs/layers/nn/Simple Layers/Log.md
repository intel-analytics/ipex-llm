## Log ##

**Scala:**
```scala
val log = Log()
```
**Python:**
```python
log = Log()
```

The Log module applies a log transformation to the input data

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val log = Log()
val input = Tensor(T(1.0f, Math.E.toFloat))
val gradOutput = Tensor(T(1.0f, 1.0f))
val output = log.forward(input)
val gradient = log.backward(input, gradOutput)
-> print(output)
0.0
1.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2]

-> print(gradient)
1.0
0.36787945
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
import math
log = Log()
input = np.array([1.0, math.e])
grad_output = np.array([1.0, 1.0])
output = log.forward(input)
gradient = log.backward(input, grad_output)

-> print output
[ 0.  1.]

-> print gradient
[ 1.          0.36787945]
```
