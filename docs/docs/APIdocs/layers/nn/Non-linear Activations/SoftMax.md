## SoftMax ##

**Scala:**
```scala
val layer = SoftMax()
```
**Python:**
```python
layer = SoftMax()
```

Applies the SoftMax function to an n-dimensional input Tensor, rescaling them so that the
elements of the n-dimensional output Tensor lie in the range (0, 1) and sum to 1.
Softmax is defined as: f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)
where shift = max_i(x_i).

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
val layer = SoftMax[Float]()
val input = Tensor[Float](3)
input.apply1(_ => 1.0f * 10)
val target = Tensor[Float](T(
1.0f,
0.0f,
0.0f
))
val output = layer.forward(input)
val gradient = layer.backward(input, target)
-> print(output)
0.33333334
0.33333334
0.33333334
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
-> print(gradient)
0.22222221
-0.11111112
-0.11111112
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
layer = SoftMax()
input = np.ones(3)*10
target = np.array([1.0, 0.0, 0.0])
output = layer.forward(input)
gradient = layer.backward(input, target)
-> print output
[ 0.33333334  0.33333334  0.33333334]
-> print gradient
[ 0.22222221 -0.11111112 -0.11111112]
```
