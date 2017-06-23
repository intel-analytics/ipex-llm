## Power ##

**Scala:**
```scala
val module = Power(power, scale, shift)
```
**Python:**
```python
module = Power(power, scale, shift)
```

 Apply an element-wise power operation with scale and shift.
 
 f(x) = (shift + scale * x)^power^
 
 `power` the exponent.
 `scale` Default is 1.
 `shift` Default is 0.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.Storage

val power = Power[Double](2, 1, 1)
val input = Tensor(Storage[Double](Array(0.0, 1, 2, 3, 4, 5)), 1, Array(2, 3))
> print(power.forward(input))
1.0	    4.0	     9.0	
16.0	    25.0     36.0	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
from bigdl.nn.layer import *

power = Power(2.0, 1.0, 1.0)
input = np.array([[0.0, 1, 2], [3, 4, 5]])
>power.forward(input)
array([[  1.,   4.,   9.],
       [ 16.,  25.,  36.]], dtype=float32)

```