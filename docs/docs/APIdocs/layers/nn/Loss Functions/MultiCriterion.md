## MultiCriterion ##

**Scala:**
```scala
val module = MultiCriterion()
```
**Python:**
```python
module = MultiCriterion()
```

MultiCriterion is a weighted sum of other criterions each applied to the same input and target

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val module = MultiCriterion()
val nll = ClassNLLCriterion()
val mse = MSECriterion()
module.add(nll, 0.5)
module.add(mse)

val input = Tensor(5).randn()
val target = Tensor(5)
target(Array(1)) = 1
target(Array(2)) = 2
target(Array(3)) = 3
target(Array(4)) = 2
target(Array(5)) = 1

val output = module.forward(input, target)

> input
1.0641425
-0.33507252
1.2345984
0.08065767
0.531199
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]


> output
res7: Float = 1.9633228
```

**Python example:**
```python
from bigdl.nn.criterion import *
import numpy as np

module = MultiCriterion()
nll = ClassNLLCriterion()
mse = MSECriterion()
module.add(nll, 0.5)
module.add(mse)

input = np.array([0.9682213801388531,
0.35258855644097503,
0.04584479998452568,
-0.21781499692588918,
-1.02721844006879])
target = np.array([1, 2, 3, 2, 1])

output = module.forward(input, target)

> output
3.6099546
```
