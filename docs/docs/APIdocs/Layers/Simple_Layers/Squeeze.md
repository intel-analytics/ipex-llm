## Squeeze ##

**Scala:**
```scala
val module = Squeeze(dims=null, numInputDims=Int.MinValue)
```
**Python:**
```python
module = Squeeze(dims, numInputDims=-2147483648)
```

Delete all singleton dimensions or a specific singleton dimension.

 `dims` Optional. If this dimension is singleton dimension, it will be deleted.
           The first index starts from 1. Default: delete all dimensions.
           
 `num_input_dims` Optional. If in a batch model, set to the inputDims.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val layer = Squeeze(2)
> print(layer.forward(Tensor(2, 1, 3).rand()))
0.43709445	0.42752415	0.43069172	
0.67029667	0.95641375	0.28823504	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
from bigdl.nn.layer import *

layer = Squeeze(2)
>layer.forward(np.array([[[1, 2, 3]], [[1, 2, 3]]]))
out: array([[ 1.,  2.,  3.],
            [ 1.,  2.,  3.]], dtype=float32)

```