## Regularizers ##

**Scala:**
```scala
val l1Regularizer = L1Regularizer(l1: Double)
val l2Regularizer = L2Regularizer(l2: Double)
val l1l2Regularizer = L1L2Regularizer(l1: Double, l2: Double)
```
**Python:**
```python
regularizerl1 = L1Regularizer(0.1)
regularizerl2 = L2Regularizer(0.1)
regularizerl1l2 = L1L2Regularizer(0.1, 0.1)
```

L1 and L2 regularizers are used to avoid overfitting.


**Scala example:**
```scala

import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.optim._

RNG.setSeed(100)

val input = Tensor[Float](3, 5).rand
val linear = Linear[Float](5, 5, wRegularizer = L2Regularizer(0.1), bRegularizer = L2Regularizer(0.1))

val output = linear.forward(input)

> println(input)
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.54340494      0.67115563      0.2783694       0.4120464       0.4245176
0.52638245      0.84477615      0.14860484      0.004718862     0.15671109
0.12156912      0.18646719      0.67074907      0.21010774      0.82585275
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x5]


> println(output)
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.34797725     0.25985366      -0.1107063      0.44529563      0.18934922
-0.36947984     0.3738199       0.033136755     0.68634266      0.31736165
-0.21293467     -0.16091438     -0.15637109     0.12860553      0.2332296
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x5]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

input = np.random.uniform(0, 1, (3, 5)).astype("float32")
linear = Linear(5, 5, wRegularizer = L2Regularizer(0.1), bRegularizer = L2Regularizer(0.1))
output = linear.forward(input)

> input
array([[ 0.14070892,  0.35661909,  0.00720507,  0.96832764,  0.34936094],
       [ 0.14347534,  0.74504513,  0.16517557,  0.27037948,  0.28448409],
       [ 0.28334993,  0.37042555,  0.32039529,  0.66894925,  0.19935906]], dtype=float32)

> output
array([[-0.482759  , -0.12087041, -0.76120645,  0.16693172, -0.21038117],
       [-0.17725618, -0.20931029, -0.53776515,  0.03298397, -0.40130591],
       [-0.36628127, -0.32192633, -0.64229649,  0.16954683, -0.15714465]], dtype=float32)
```