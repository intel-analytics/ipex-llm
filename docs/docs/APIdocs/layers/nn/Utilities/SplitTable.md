## SplitTable ##

**Scala:**
```scala
val layer = SplitTable[T](dim)
```
**Python:**
```python
layer = SplitTable(dim)
```

SplitTable takes a Tensor as input and outputs several tables,
splitting the Tensor along the specified dimension `dimension`. Please note
the dimension starts from 1.

The input to this layer is expected to be a tensor, or a batch of tensors;
when using mini-batch, a batch of sample tensors will be passed to the layer and
the user needs to specify the number of dimensions of each sample tensor in a
batch using `nInputDims`.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor

val layer = SplitTable[Float](2)
layer.forward(Tensor(T(
  T(1.0f, 2.0f, 3.0f),
  T(4.0f, 5.0f, 6.0f),
  T(7.0f, 8.0f, 9.0f)
)))
layer.backward(Tensor(T(
  T(1.0f, 2.0f, 3.0f),
  T(4.0f, 5.0f, 6.0f),
  T(7.0f, 8.0f, 9.0f)
)), T(
  Tensor[Float](T(0.1f, 0.2f, 0.3f)),
  Tensor[Float](T(0.4f, 0.5f, 0.6f)),
  Tensor[Float](T(0.7f, 0.8f, 0.9f))
))
```

Its output should be 
```
 {
        2: 2.0
           5.0
           8.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
        1: 1.0
           4.0
           7.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
        3: 3.0
           6.0
           9.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
 }

0.1     0.4     0.7
0.2     0.5     0.8
0.3     0.6     0.9
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**
```python
from bigdl.nn.layer import SplitTable
import numpy as np

layer = SplitTable(2)
layer.forward(np.array([
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0],
  [7.0, 8.0, 9.0]
]))

layer.backward(np.array([
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0],
  [7.0, 8.0, 9.0]
]), [
  np.array([0.1, 0.2, 0.3]),
  np.array([0.4, 0.5, 0.6]),
  np.array([0.7, 0.8, 0.9])
])
```

Its output should be
```
[
  array([ 1.,  4.,  7.], dtype=float32),
  array([ 2.,  5.,  8.], dtype=float32),
  array([ 3.,  6.,  9.], dtype=float32)
]

array([[ 0.1       ,  0.40000001,  0.69999999],
       [ 0.2       ,  0.5       ,  0.80000001],
       [ 0.30000001,  0.60000002,  0.89999998]], dtype=float32)
```