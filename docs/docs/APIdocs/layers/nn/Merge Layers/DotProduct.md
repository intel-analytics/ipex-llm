## DotProduct ##
outputs the dot product (similarity) between inputs


**Scala:**

```scala
DotProduct[T]()
```
**Python:**
```python
DotProduct(bigdl_type="float")
```


**Scala example:**
```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.{T, Table}

val mlp = DotProduct[Double]()
val x = Tensor[Double](2, 3).fill(1)
val y = Tensor[Double](2, 3).fill(2)
print(mlp.forward(T(x, y)))
```
```
6.0
6.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
```

**Python example:**

```python
import numpy as np
from bigdl.nn.layer import *

mlp = DotProduct()
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(mlp.forward([x, y]))
print(mlp.forward([np.ones((2, 3)), np.full((2, 3), 2)]))

```
```
output is:
[array([ 32.], dtype=float32)]
[array([ 6.,  6.], dtype=float32)]
```

