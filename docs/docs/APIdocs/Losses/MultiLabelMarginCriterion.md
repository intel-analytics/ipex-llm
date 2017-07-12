## MultiLabelMarginCriterion ##

**Scala:**
```scala
val multiLabelMarginCriterion = MultiLabelMarginCriterion(sizeAverage = true)
```
**Python:**
```python
multiLabelMarginCriterion = MultiLabelMarginCriterion(size_average=True)
```
MultiLabelMarginCriterion creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss) between input x and output y 

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val multiLabelMarginCriterion = MultiLabelMarginCriterion(false)
val input = Tensor(4).rand()
val target = Tensor(4)
target(Array(1)) = 3
target(Array(2)) = 2
target(Array(3)) = 1
target(Array(4)) = 0

> print(input)
0.40267515
0.5913795
0.84936756
0.05999674

>  print(multiLabelMarginCriterion.forward(input, target))
0.33414197

> print(multiLabelMarginCriterion.backward(input, target))
-0.25
-0.25
-0.25
0.75
[com.intel.analytics.bigdl.tensor.DenseTensor of size 4]


```

**Python example:**
```python
from bigdl.nn.layer import *
multiLabelMarginCriterion = MultiLabelMarginCriterion(False)

> multiLabelMarginCriterion.forward(np.array([0.3, 0.4, 0.2, 0.6]), np.array([3, 2, 1, 0]))
0.975

> multiLabelMarginCriterion.backward(np.array([0.3, 0.4, 0.2, 0.6]), np.array([3, 2, 1, 0]))
[array([-0.25, -0.25, -0.25,  0.75], dtype=float32)]

```

