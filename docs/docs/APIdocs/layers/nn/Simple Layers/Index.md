## Index ##

**Scala:**
```scala
val model = Index(dimension)
```
**Python:**
```python
model = Index(dimension)
```

Applies the Tensor index operation along the given dimension.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

val input1 = Tensor(3).rand()
val input2 = Tensor(4)
input2(Array(1)) = 1.0f
input2(Array(2)) = 2.0f
input2(Array(3)) = 2.0f
input2(Array(4)) = 3.0f

val input = T(input1, input2)
val model = Index(1)
val output = model.forward(input)

scala> print(input)
 {
	2: 1.0
	   2.0
	   2.0
	   3.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 4]
	1: 0.124325536
	   0.8768922
	   0.6378146
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
 }
scala> print(output)
0.124325536
0.8768922
0.8768922
0.6378146
[com.intel.analytics.bigdl.tensor.DenseTensor of size 4]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np
 
input1 = np.random.randn(3)
input2 = np.array([1, 2, 2, 3])
input = [input1, input2]

model = Index(1)
output = model.forward(input)

>>> print(input)
[array([-0.45804847, -0.20176707,  0.50963248]), array([1, 2, 2, 3])]

>>> print(output)
[-0.45804846 -0.20176707 -0.20176707  0.50963247]
```
