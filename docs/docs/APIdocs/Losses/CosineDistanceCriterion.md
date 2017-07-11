## CosineDistanceCriterion ##

**Scala:**
```scala
val criterion = CosineDistanceCriterion()
```
**Python:**
```python
criterion = CosineDistanceCriterion()
```

 This loss function measures the Cosine Distance between the target and the output
``` 
 loss(o, t) = 1 - cos(o, t)
```
 By default, the losses are averaged for each mini-batch over observations as well as over
 dimensions. However, if the field sizeAverage is set to false, the losses are instead summed.

**Scala example:**
```scala
import com.intel.analytics.bigdl.numeric.NumericFloat

val criterion = CosineDistanceCriterion()
val input = Tensor(1, 5).rand
val target = Tensor(1, 5).rand
val loss = criterion.forward(input, target)

> println(target)
target: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.95363826	0.3175587	0.90366143	0.10316128	0.05317958
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5]


> println(input)
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.5895327	0.20547494	0.43118918	0.28824696	0.032088008
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5]


> println(loss)
loss: Float = 0.048458755

```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

criterion = CosineDistanceCriterion()
input = np.random.uniform(0, 1, (5, 1)).astype("float32")
target = np.random.uniform(0, 1, (5, 1)).astype("float32")
loss = criterion.forward(input, target)

> input
array([[ 0.92156637],
       [ 0.72886127],
       [ 0.49714726],
       [ 0.74645835],
       [ 0.16812921]], dtype=float32)

> target 
array([[ 0.83094525],
       [ 0.95740199],
       [ 0.86036712],
       [ 0.84625793],
       [ 0.0625345 ]], dtype=float32)

> loss
-1.1920929e-08

```
