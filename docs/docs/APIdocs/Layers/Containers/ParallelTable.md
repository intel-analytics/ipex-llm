## ParallelTable ##

**Scala:**
```scala
val module = ParallelTable()
```
**Python:**
```python
module = ParallelTable()
```

It is a container module that applies the i-th member module to the i-th
 input, and outputs an output in the form of Table
 
**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T 
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val module = ParallelTable()
val log = Log()
val exp = Exp()
module.add(log)
module.add(exp)
val input1 = Tensor(3, 3).rand(0, 1)
val input2 = Tensor(3).rand(0, 1)
val input = T(1 -> input1, 2 -> input2)
> print(module.forward(input))
 {
        2: 2.6996834
           2.0741253
           1.0625387
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
        1: -1.073425    -0.6672964      -1.8160943
           -0.54094607  -1.3029919      -1.7064717
           -0.66175103  -0.08288143     -1.1840979
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
 }
```

**Python example:**
```python
from bigdl.nn.layer import *

module = ParallelTable()
log = Log()
exp = Exp()
module.add(log)
module.add(exp)
input1 = np.random.rand(3,3)
input2 = np.random.rand(3)
>module.forward([input1, input2])
[array([[-1.27472472, -2.18216252, -0.60752904],
        [-2.76213861, -1.77966928, -0.13652121],
        [-1.47725129, -0.03578046, -1.37736678]], dtype=float32),
 array([ 1.10634041,  1.46384597,  1.96525407], dtype=float32)]
```