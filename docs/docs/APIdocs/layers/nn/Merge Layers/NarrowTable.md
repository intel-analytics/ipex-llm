## NarrowTable ##

**Scala:**
```scala
val narrowTable = NarrowTable(var offset: Int, val length: Int = 1)
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
val narrowTable = NarrowTable(1, 1)

val input = T()
input(1.0) = Tensor(1, 1).rand()
input(2.0) = Tensor(2, 2).rand()
input(3.0) = Tensor(3, 3).rand()

> print(narrowTable.forward(input))
 {
	1: 0.8369138888083398	
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 1x1]
 }

```

**Python example:**
```python
from bigdl.nn.layer import *
narrowTable = NarrowTable(1, 1)
> narrowTable.forward([np.array([1, 2, 3]), np.array([4, 5, 6])])
[array([ 1.,  2.,  3.], dtype=float32)]
       
```

