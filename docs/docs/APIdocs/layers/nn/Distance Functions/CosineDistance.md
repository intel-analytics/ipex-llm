## CosineDistance ##

**Scala:**
```scala
val module = CosineDistance()
```
**Python:**
```python
module = CosineDistance()
```

CosineDistance creates a module that takes a table of two vectors (or matrices if in batch mode) as input and outputs the cosine distance between them.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val module = CosineDistance()
val t1 = Tensor().range(1, 3)
val t2 = Tensor().range(4, 6)
val input = T(t1, t2)
val output = module.forward(input)

> input
input: com.intel.analytics.bigdl.utils.Table =
 {
	2: 4.0
	   5.0
	   6.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
	1: 1.0
	   2.0
	   3.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
 }

> output
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.9746319
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = CosineDistance()
t1 = np.array([1.0, 2.0, 3.0])
t2 = np.array([4.0, 5.0, 6.0])
input = [t1, t2]
output = module.forward(input)

> input
[array([ 1.,  2.,  3.]), array([ 4.,  5.,  6.])]

> output
[ 0.97463191]
```
