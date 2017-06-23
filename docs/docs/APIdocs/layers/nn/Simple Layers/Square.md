## Square ##

**Scala:**
```scala
val module = Square()
```
**Python:**
```python
module = Square()
```

Square apply an element-wise square operation.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val module = Square()

println(module.forward(Tensor.range(1, 6, 1)))
```
Output is
```
com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0
4.0
9.0
16.0
25.0
36.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 6]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Square()
print(module.forward(np.arange(1, 7, 1)))
```
Output is
```
[array([  1.,   4.,   9.,  16.,  25.,  36.], dtype=float32)]
```
