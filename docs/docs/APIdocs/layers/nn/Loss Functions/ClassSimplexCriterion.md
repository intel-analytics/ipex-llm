## ClassSimplexCriterion ##

**Scala:**
```scala
val criterion = ClassSimplexCriterion(nClasses)
```
**Python:**
```python
criterion = ClassSimplexCriterion(nClasses)
```

ClassSimplexCriterion implements a criterion for classification.
It learns an embedding per class, where each class' embedding is a
point on an (N-1)-dimensional simplex, where N is the number of classes.

**Parameters:**
* **nClasses** - An integer, the number of classes.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
val criterion = ClassSimplexCriterion[Float](5)
val input = Tensor[Float](T(
 T(1.0f, 2.0f, 3.0f, 4.0f, 5.0f),
 T(4.0f, 5.0f, 6.0f, 7.0f, 8.0f)
))
val target = Tensor[Float](2)
target(Array(1)) = 2.0f
target(Array(2)) = 1.0f
val output = criterion.forward(input, target)
val gradient = criterion.backward(input, target)
-> print(output)
23.562702
-> print(gradient)
0.25    0.20635083      0.6     0.8     1.0     
0.6     1.0     1.2     1.4     1.6     
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x5]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
criterion = ClassSimplexCriterion(5)
input = np.array([
   [1.0, 2.0, 3.0, 4.0, 5.0],
   [4.0, 5.0, 6.0, 7.0, 8.0]
])
target = np.array([2.0, 1.0])
output = criterion.forward(input, target)
gradient = criterion.backward(input, target)
-> print output
23.562702
-> print gradient
[[ 0.25        0.20635083  0.60000002  0.80000001  1.        ]
 [ 0.60000002  1.          1.20000005  1.39999998  1.60000002]]
```
