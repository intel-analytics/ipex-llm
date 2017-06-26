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

val input1Arr = Array(1, 2, 3, 4, 5)
val input2Arr = Array(5, 4, 3, 2, 1)

val target1Arr = Array(-1, 1, -1, 1, 1)

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
output: Float = 0.8                                                                                                                                                                    [21/154]
```

The gradInput will be,

```
gradInput: com.intel.analytics.bigdl.utils.Table =
 {
        2: -0.0
           0.2
           -0.2
           0.0
           0.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
        1: 0.0
           -0.2
           0.2
           -0.0
           -0.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
 }
```

**Python example:**

```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

mse = MarginRankingCriterion()

input1 = np.array([1, 2, 3, 4, 5]).astype("float32")
input2 = np.array([5, 4, 3, 2, 1]).astype("float32")
input = [input1, input2]

target1 = np.array([-1, 1, -1, 1, 1]).astype("float32")
target = [target1, target1]

output = mse.forward(input, target)
gradInput = mse.backward(input, target)

print output
print gradInput
```

The output will be,

```
0.8
```

The gradInput will be,

```
[array([ 0. , -0.2,  0.2, -0. , -0. ], dtype=float32), array([-0. ,  0.2, -0.2,  0. ,  0. ], dtype=float32)] 
```

