## NarrowTable ##

**Scala:**
```scala
val narrowTable = NarrowTable(offset, length = 1)
```
**Python:**
```python
narrowTable = NarrowTable(offset, length = 1)
```

NarrowTable takes a table as input and returns a subtable starting from index `offset` having `length` elements

Negative `length` means the last element is located at Abs|length| to the last element of input

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T
val narrowTable = NarrowTable(1, 1)

val input = T()
input(1.0) = Tensor(2, 2).rand()
input(2.0) = Tensor(2, 2).rand()
input(3.0) = Tensor(2, 2).rand()
> print(input)
 {
	2.0: 0.27686104	0.9040761	
	     0.75969505	0.8008061	
	     [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
	1.0: 0.94122535	0.46173728	
	     0.43302807	0.1670979	
	     [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
	3.0: 0.43944374	0.49336782	
	     0.7274511	0.67777634	
	     [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
 }
>  print(narrowTable.forward(input))
 {
	1: 0.94122535	0.46173728	
	   0.43302807	0.1670979	
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
 }

```

**Python example:**
```python
from bigdl.nn.layer import *
narrowTable = NarrowTable(1, 1)
> narrowTable.forward([np.array([1, 2, 3]), np.array([4, 5, 6])])
[array([ 1.,  2.,  3.], dtype=float32)]
       
```

