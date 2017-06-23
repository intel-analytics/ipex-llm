## MarginRankingCriterion ##

**Scala:**

```scala
val mse = new MarginRankingCriterion()
```

**Python:**

```python
mse = MarginRankingCriterion()
```

Creates a criterion that measures the loss given an input x = {x1, x2},
a table of two Tensors of size 1 (they contain only scalars), and a label y (1 or -1).
In batch mode, x is a table of two Tensors of size batchsize, and y is a Tensor of size
batchsize containing 1 or -1 for each corresponding pair of elements in the input Tensor.
If y == 1 then it assumed the first input should be ranked higher (have a larger value) than
the second input, and vice-versa for y == -1.


**Scala example:**

```scala
import com.intel.analytics.bigdl.nn.MarginRankingCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.Storage
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

import scala.util.Random

val input1Arr = Array(0.20028763,  0.27957329,  0.05694006,  0.38143176,  0.02489306)
val input2Arr = Array(0.20028763,  0.47957329,  0.05694006,  0.58143176,  0.02489306)

val target1Arr = Array(0.10028763,  0.27957329,  0.05694006,  0.38143176,  0.12489306)

val input1 = Tensor(Storage(input1Arr.map(x => x.toFloat)))
val input2 = Tensor(Storage(input2Arr.map(x => x.toFloat)))

val input = T((1.toFloat, input1), (2.toFloat, input2))

val target1 = Tensor(Storage(target1Arr.map(x => x.toFloat)))
val target = T((1.toFloat, target1))

val mse = new MarginRankingCriterion()

val output = mse.forward(input, target)
val gradInput = mse.backward(input, target)

println(output)
println(gradInput)
```

The output will be,

```
1.0264403

```

The gradInput will be,

```
 {
        2: 0.020057527
           0.05591466
           0.011388012
           0.07628635
           0.024978613
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
        1: -0.020057527
           -0.05591466
           -0.011388012
           -0.07628635
           -0.024978613
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
 }

```

**Python example:**

```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

mrc = MarginRankingCriterion()

input1 = np.array([0.20028763,  0.27957329,  0.05694006,  0.38143176,  0.02489306]).astype("float32")
input2 = np.array([0.20028763,  0.47957329,  0.05694006,  0.58143176,  0.02489306]).astype("float32")
input = [input1, input2]

target1 = np.array([0.10028763,  0.27957329,  0.05694006,  0.38143176,  0.12489306]).astype("float32")
target = [target1]

output = mrc.forward(input, target)
gradInput = mrc.backward(input, target)

print "output = \n %s" % output
print "gradInput = \n %s " % gradInput
```

The output will be,

```
output = 
 1.0264403
gradInput = 
 [array([-0.02005753, -0.05591466, -0.01138801, -0.07628635, -0.02497861], dtype=float32), array([ 0.02005753,  0.05591466,  0.01138801,  0.07628635,  0.02497861], dtype=float32)] 
```

