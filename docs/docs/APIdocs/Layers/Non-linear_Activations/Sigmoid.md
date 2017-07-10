## Sigmoid ##

**Scala:**
```scala
val module = Sigmoid()
```
**Python:**
```python
module = Sigmoid()
```

Applies the Sigmoid function element-wise to the input Tensor,
thus outputting a Tensor of the same dimension.

Sigmoid is defined as: f(x) = 1 / (1 + exp(-x))
  

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val layer = new Sigmoid()
val input = Tensor(2, 3)
var i = 0
input.apply1(_ => {i += 1; i})
> print(layer.forward(input))
0.7310586	0.880797	0.95257413	
0.98201376	0.9933072	0.9975274	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
from bigdl.nn.layer import *

layer = Sigmoid()
input = np.array([[1, 2, 3], [4, 5, 6]])
>layer.forward(input)
array([[ 0.7310586 ,  0.88079703,  0.95257413],
       [ 0.98201376,  0.99330717,  0.99752742]], dtype=float32)
```