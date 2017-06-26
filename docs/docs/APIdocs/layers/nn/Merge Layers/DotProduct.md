## DotProduct ##

**Scala:**

```scala
DotProduct[T]()
```
**Python:**
```python
DotProduct(bigdl_type="float")
```

Outputs the dot product (similarity) between inputs


**Scala example:**
```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.{T, Table}

val mlp = DotProduct[Double]()
val x = Tensor[Double](3).fill(1)
val y = Tensor[Double](3).fill(2)
println("input:")
println(x)
println(y)
println("output:")
println(mlp.forward(T(x, y)))
```
```
input:
1.0
1.0
1.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
2.0
2.0
2.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
output:
6.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1]

```

**Python example:**

```python
import numpy as np
from bigdl.nn.layer import *

mlp = DotProduct()
x = np.array([1, 1, 1])
y = np.array([2, 2, 2])
print("input:")
print(x)
print(y)
print("output:")
print(mlp.forward([x, y]))

```
```
creating: createDotProduct
input:
[1 1 1]
[2 2 2]
output:
[ 6.]
```

