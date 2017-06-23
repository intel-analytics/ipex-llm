## Squeeze ##

**Scala:**
```scala
val module = Squeeze(dims, batchMode)
```
**Python:**
```python
module = Squeeze(dims, batchMode)
```

Delete all singleton dimensions or a specific singleton dimension.

 `dims` Optional. If this dimension is singleton dimension, it will be deleted.
           The first index starts from 1. Default: delete all dimensions.
           
 `batchMode` Optional. If the input is batch. Default is false.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

val layer = Squeeze[Float](2)
> print(layer.forward(Tensor[Float](2, 1, 3).rand()))
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