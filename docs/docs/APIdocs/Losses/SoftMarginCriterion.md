## SoftMarginCriterion ##

**Scala:**
```scala
val criterion = SoftMarginCriterion(sizeAverage)
```
**Python:**
```python
criterion = SoftMarginCriterion(size_average)
```

Creates a criterion that optimizes a two-class classification logistic loss between
input x (a Tensor of dimension 1) and output y (which is a tensor containing either
1s or -1s).
```
loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x:nElement()
```

**Parameters:**
* **sizeAverage** - A boolean indicating whether normalizing by the number of elements in the input.
                    Default: true

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val criterion = SoftMarginCriterion()
val input = Tensor(T(
 T(1.0f, 2.0f),
 T(3.0f, 4.0f))
)
val target = Tensor(T(
 T(1.0f, -1.0f),
 T(-1.0f, 1.0f))
)
val output = criterion.forward(input, target)
val gradient = criterion.backward(input, target)
-> print(output)
1.3767318
-> print(gradient)
-0.06723536     0.22019927      
0.23814353      -0.0044965525   
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
criterion = SoftMarginCriterion()
input = np.array([
          [1.0, 2.0],
          [3.0, 4.0]
        ])
target = np.array([
           [2.0, 3.0],
           [4.0, 5.0]
         ])
output = criterion.forward(input, target)
gradient = criterion.backward(input, target)
-> print output
1.3767318
-> print gradient
[[-0.06723536  0.22019927]
 [ 0.23814353 -0.00449655]]
```
