## MultiLabelSoftMarginCriterion ##

**Scala:**
```scala
val criterion = MultiLabelSoftMarginCriterion(weights = null, sizeAverage = true)
```
**Python:**
```python
criterion = MultiLabelSoftMarginCriterion(weights=None, size_average=True)
```

MultiLabelSoftMarginCriterion is a multiLabel multiclass criterion based on sigmoid:
```
l(x,y) = - sum_i y[i] * log(p[i]) + (1 - y[i]) * log (1 - p[i])
```
 where ```p[i] = exp(x[i]) / (1 + exp(x[i]))```
 
 If with weights,
 ```
 l(x,y) = - sum_i weights[i] (y[i] * log(p[i]) + (1 - y[i]) * log (1 - p[i]))
 ```

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val criterion = MultiLabelSoftMarginCriterion()
val input = Tensor(3)
input(Array(1)) = 0.4f
input(Array(2)) = 0.5f
input(Array(3)) = 0.6f
val target = Tensor(3)
target(Array(1)) = 0
target(Array(2)) = 1
target(Array(3)) = 1

> criterion.forward(input, target)
res0: Float = 0.6081934
```

**Python example:**
```python
from bigdl.nn.criterion import *
import numpy as np

criterion = MultiLabelSoftMarginCriterion()
input = np.array([0.4, 0.5, 0.6])
target = np.array([0, 1, 1])

> criterion.forward(input, target)
0.6081934
```
