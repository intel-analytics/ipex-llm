## CMinTable ##

**Scala:**
```scala
val layer = CMinTable[T]()
```
**Python:**
```python
layer = CMinTable()
```

CMinTable takes a bunch of tensors as inputs. These tensors must have
same shape. This layer will merge them by doing an element-wise comparision
and use the min value.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor

val layer = CMinTable[Float]()
layer.forward(T(
  Tensor[Float](T(1.0f, 5.0f, 2.0f)),
  Tensor[Float](T(3.0f, 4.0f, -1.0f)),
  Tensor[Float](T(5.0f, 7.0f, -5.0f))
))
layer.backward(T(
  Tensor[Float](T(1.0f, 5.0f, 2.0f)),
  Tensor[Float](T(3.0f, 4.0f, -1.0f)),
  Tensor[Float](T(5.0f, 7.0f, -5.0f))
), Tensor[Float](T(0.1f, 0.2f, 0.3f)))
```
Its output should be
```
1.0
4.0
-5.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]

{
  2: 0.0
     0.2
     0.0
     [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
  1: 0.1
     0.0
     0.0
     [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
  3: 0.0
     0.0
     0.3
  [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
}
```

**Python example:**
```python
from bigdl.nn.layer import CMinTable
import numpy as np

layer = CMinTable()
layer.forward([
  np.array([1.0, 5.0, 2.0]),
  np.array([3.0, 4.0, -1.0]),
  np.array([5.0, 7.0, -5.0])
])

layer.backward([
  np.array([1.0, 5.0, 2.0]),
  np.array([3.0, 4.0, -1.0]),
  np.array([5.0, 7.0, -5.0])
], np.array([0.1, 0.2, 0.3]))

```
Its output should be
```
array([ 1.,  4., -5.], dtype=float32)

[array([ 0.1, 0., 0.], dtype=float32),
array([ 0., 0.2, 0.], dtype=float32),
array([ 0., 0., 0.30000001], dtype=float32)]

```