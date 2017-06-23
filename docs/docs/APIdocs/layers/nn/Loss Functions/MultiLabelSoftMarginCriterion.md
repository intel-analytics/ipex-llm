## MultiLabelSoftMarginCriterion ##

**Scala:**
```scala
val module = MultiLabelSoftMarginCriterion(weights: Tensor[T] = null, sizeAverage: Boolean = true)
```
**Python:**
```python
module = MultiLabelSoftMarginCriterion(weights=None, size_average=True)
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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericDouble

val criterion = MultiLabelSoftMarginCriterion()
val input = Tensor(3)
input(Array(1)) = 0.4
input(Array(2)) = 0.5
input(Array(3)) = 0.6
val target = Tensor(3)
target(Array(1)) = 0
target(Array(2)) = 1
target(Array(3)) = 1

> criterion.forward(input, target)
res0: Double = 0.608193395686766
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
