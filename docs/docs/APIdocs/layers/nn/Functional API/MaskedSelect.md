## MaskedSelect ##

**Scala:**
```scala
val module = MaskedSelect()
```
**Python:**
```python
module = MaskedSelect()
```

Performs a torch.MaskedSelect on a Tensor. The mask is supplied as a tabular argument
 with the input on the forward and backward passes.
 
**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import scala.util.Random


val layer = MaskedSelect()
val input1 = Tensor(2, 2).apply1(e => Random.nextFloat())
val mask = Tensor(2, 2)
mask(Array(1, 1)) = 1
mask(Array(1, 2)) = 0
mask(Array(2, 1)) = 0
mask(Array(2, 2)) = 1
val input = T()
input(1.0) = input1
input(2.0) = mask
> print(layer.forward(input))
0.2577119
0.5061479
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
```

**Python example:**
```python
from bigdl.nn.layer import *

layer = MaskedSelect()
input1 = np.random.rand(2,2)
mask = np.array([[1,0], [0, 1]])
>layer.forward([input1, mask])
array([ 0.1525335 ,  0.05474588], dtype=float32)
```