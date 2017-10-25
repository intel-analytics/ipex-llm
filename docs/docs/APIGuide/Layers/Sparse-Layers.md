
---
## SparseLinear ##

**Scala:**
```scala
val module = SparseLinear(
  inputSize,
  outputSize,
  withBias = true,
  backwardStart: Int = -1,
  backwardLength: Int = -1,
  wRegularizer = null,
  bRegularizer = null,
  initWeight = null,
  initBias = null,
  initGradWeight = null,
  initGradBias = null)
```
**Python:**
```python
module = SparseLinear(
  input_size,
  output_size,
  init_method="default",
  with_bias=True,
  backwardStart=-1,
  backwardLength=-1,
  wRegularizer=None,
  bRegularizer=None,
  init_weight=None,
  init_bias=None,
  init_grad_weight=None,
  init_grad_bias=None)
```

SparseLinear is the sparse version of module Linear. SparseLinear has two different from Linear: firstly, SparseLinear's input Tensor is a SparseTensor. Secondly, SparseLinear doesn't backward gradient to next layer in the backpropagation by default, as the gradInput of SparseLinear is useless and very big in most cases.

But, considering model like Wide&Deep, we provide backwardStart and backwardLength to backward part of the gradient to next layer.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val module = SparseLinear(1000, 5)

val input = Tensor.sparse(Array(Array(0, 0, 0, 1, 1, 1), Array(1, 5, 300, 2, 100, 500)), Array(1f, 3f, 5f, 2f, 4f, 6f), Array(2, 1000))

println(module.forward(input))
```

Gives the output,
```
scala> println(module.forward(input))
0.047791008	0.069045454	0.020120896	0.019826084	0.10610865	
-0.059406646	-0.13536823	-0.13861635	0.070304416	0.009570055	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x5]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.common import *
import numpy as np

module = SparseLinear(1000, 5)

input = JTensor.sparse(np.array(1, 3, 5, 2, 4, 6), np.array(0, 0, 0, 1, 1, 1, 1, 5, 300, 2, 100, 500), np.array(1000, 5))

print(module.forward(input)))
```
Gives the output,
```
[[ 10.09569263 -10.94844246  -4.1086688    1.02527523  11.80737209]
 [  7.9651413    9.7131443  -10.22719955   0.02345783  -3.74368906]]
```
---
