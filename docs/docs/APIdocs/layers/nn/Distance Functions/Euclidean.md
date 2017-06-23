## Euclidean ##

**Scala:**
```scala
val module = Euclidean(
  inputSize: Int,
  outputSize: Int,
  fastBackward: Boolean = true)
```
**Python:**
```python
module = Euclidean(
  input_size,
  output_size,
  fast_backward=True)
```
Outputs the Euclidean distance of the input to `outputSize` centers.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val module = Euclidean(3, 3)

println(module.forward(Tensor.range(1, 3, 1)))
```
Output is
```
com.intel.analytics.bigdl.tensor.Tensor[Float] =
4.0323668
3.7177157
3.8736997
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Euclidean(3, 3)

print(module.forward(np.arange(1, 4, 1)))
```
Output is
```
[array([ 3.86203027,  4.02212906,  3.2648952 ], dtype=float32)]
```
