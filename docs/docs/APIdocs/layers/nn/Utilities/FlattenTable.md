## FlattenTable ##

**Scala:**
```scala
val module = FlattenTable()
```
**Python:**
```python
module = FlattenTable()
```

FlattenTable takes an arbitrarily deep table of Tensors (potentially nested) as input and a table of Tensors without any nested table will be produced

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericDouble

val module = FlattenTable()
val t1 = Tensor(3).randn()
val t2 = Tensor(3).randn()
val t3 = Tensor(3).randn()
val input = T(t1, T(t2, T(t3)))

val output = module.forward(input)

> input
 {
	2:  {
	   	2:  {
	   	   	1: -0.7738335778343488
	   	   	   1.0884042854505709
	   	   	   -1.0361592723999347
	   	   	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]
	   	    }
	   	1: -5.671122490419898E-4
	   	   -0.0464522284021047
	   	   0.391028022141935
	   	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]
	    }
	1: 0.3199535448955691
	   1.4756887991498508
	   -1.0647405816201285
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]
 }

> output
{
	2: -5.671122490419898E-4
	   -0.0464522284021047
	   0.391028022141935
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]
	1: 0.3199535448955691
	   1.4756887991498508
	   -1.0647405816201285
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]
	3: -0.7738335778343488
	   1.0884042854505709
	   -1.0361592723999347
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]
 }

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Sequential()
# this will create a nested table
nested = ConcatTable().add(Identity()).add(Identity())
module.add(nested).add(FlattenTable())
t1 = np.random.randn(3)
t2 = np.random.randn(3)
input = [t1, t2]
output = module.forward(input)

> input
[array([-2.21080689, -0.48928043, -0.26122161]), array([-0.8499716 ,  1.63694575, -0.31109292])]

> output
[array([-2.21080685, -0.48928043, -0.26122162], dtype=float32),
 array([-0.84997159,  1.63694572, -0.31109291], dtype=float32),
 array([-2.21080685, -0.48928043, -0.26122162], dtype=float32),
 array([-0.84997159,  1.63694572, -0.31109291], dtype=float32)]

```
