## Select ##

**Scala:**
```scala
val layer = Select[T](dim, index)
```
**Python:**
```python
layer = Select(dim, index)
```

A Simple layer selecting an index of the input tensor in the given dimension.
Please note that the index and dimension start from 1.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor

val layer = Select[Float](1, 2)
layer.forward(Tensor[Float](T(
  T(1.0f, 2.0f, 3.0f),
  T(4.0f, 5.0f, 6.0f),
  T(7.0f, 8.0f, 9.0f)
)))

layer.backward(Tensor[Float](T(
  T(1.0f, 2.0f, 3.0f),
  T(4.0f, 5.0f, 6.0f),
  T(7.0f, 8.0f, 9.0f)
)), Tensor[Float](T(0.1f, 0.2f, 0.3f)))
```

Its output should be
```
4.0
5.0
6.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]

0.0     0.0     0.0
0.1     0.2     0.3
0.0     0.0     0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**
```python
from bigdl.nn.layer import Select
import numpy as np

layer = Select(1, 2)
layer.forward(np.array([
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0],
  [7.0, 8.0, 9.0]
]))
layer.backward(np.array([
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0],
  [7.0, 8.0, 9.0]
]), np.array([0.1, 0.2, 0.3]))
```

Its output should be
```
array([ 4.,  5.,  6.], dtype=float32)

array([[ 0.        ,  0.        ,  0.        ],
       [ 0.1       ,  0.2       ,  0.30000001],
       [ 0.        ,  0.        ,  0.        ]], dtype=float32)
```