## CMul ##

**Scala:**
```scala
val module = CMul(size, wRegularizer = null)
```
**Python:**
```python
module = CMul(size, wRegularizer=None)
```

This layer has a weight tensor with given size. The weight will be multiplied element wise to
the input tensor. If the element number of the weight tensor match the input tensor, a simply
element wise multiply will be done. Or the bias will be expanded to the same size of the input.
The expand means repeat on unmatched singleton dimension(if some unmatched dimension isn't
singleton dimension, it will report an error). If the input is a batch, a singleton dimension
will be add to the first dimension before the expand.

  `size` the size of the bias, which is an array of bias shape
  

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val layer = CMul(Array(2, 1))
val input = Tensor(2, 3)
var i = 0
input.apply1(_ => {i += 1; i})
> print(layer.forward(input))
-0.29362988     -0.58725977     -0.88088965
1.9482219       2.4352775       2.9223328
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
from bigdl.nn.layer import *

layer = CMul([2,1])
input = np.array([[1, 2, 3], [4, 5, 6]])
>layer.forward(input)
array([[-0.17618844, -0.35237688, -0.52856529],
       [ 0.85603124,  1.07003903,  1.28404689]], dtype=float32)
```