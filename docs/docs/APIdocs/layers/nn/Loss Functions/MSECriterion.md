## MSECriterion ##

**Scala:**
```scala
val criterion = MSECriterion[Float]()
```
**Python:**
```python
criterion = MSECriterion()
```

The mean squared error criterion e.g. input: a, target: b, total elements: n
```
loss(a, b) = 1/n * sum(|a_i - b_i|^2)
```

**Parameters:**

 * **sizeAverage** - a boolean indicating whether to divide the sum of squared error by n.
 Default: true

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
val criterion = MSECriterion[Float]()
val input = Tensor[Float](T(
 T(1.0f, 2.0f),
 T(3.0f, 4.0f))
)
val target = Tensor[Float](T(
 T(2.0f, 3.0f),
 T(4.0f, 5.0f))
)
val output = criterion.forward(input, target)
val gradient = criterion.backward(input, target)
-> print(output)
1.0
-> print(gradient)
-0.5    -0.5    
-0.5    -0.5    
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
criterion = MSECriterion()
input = np.array([
          [1.0, 2.0],
          [3.0, 4.0]
        ])
target = np.array([
           [2.0, 3.0],
           [4.0, 5.0]
         ])
output = criterion.forward(input, target)
gradient= criterion.backward(input, target)
-> print output
1.0
-> print gradient
[[-0.5 -0.5]
 [-0.5 -0.5]]
```
