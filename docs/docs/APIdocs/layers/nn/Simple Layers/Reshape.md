## Reshape ##

**Scala:**
```scala
val reshape = Reshape(size, batchMode)
```
**Python:**
```python
reshape = Reshape(size, batch_mode)
```

The `forward(input)` reshape the input tensor into `size(0) * size(1) * ...` tensor,
taking the elements row-wise.

**parameters**
* **size** - the size after reshape
* **batchMode** - It is a optional argument. If it is set to `Some(true)`,
                  the first dimension of input is considered as batch dimension,
                  and thus keep this dimension size fixed. This is necessary
                  when dealing with batch sizes of one. When set to `Some(false)`,
                  it forces the entire input (including the first dimension) to be reshaped
                  to the input size. Default is `None`, which means the module considers
                  inputs with more elements than the product of provided sizes (size(0) *
                  size(1) * ..) to be batches, otherwise in no batch mode.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val reshape = Reshape(Array(3, 2))
val input = Tensor(2, 2, 3).rand()
val output = reshape.forward(input)
-> print(output.size().toList)      
List(2, 3, 2)
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
reshape =  Reshape([3, 2])
input = np.random.rand(2, 2, 3)
output = reshape.forward(input)
-> print output[0].shape
(2, 3, 2)
```
