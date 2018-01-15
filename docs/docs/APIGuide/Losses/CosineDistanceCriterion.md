## CosineDistanceCriterion ##

**Scala:**
```scala
val criterion = CosineDistanceCriterion()
```
**Python:**
```python
criterion = CosineDistanceCriterion(size_average = True)
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

criterion = CosineDistanceCriterion(size_average = True)
input = np.random.uniform(0, 1, (1, 5)).astype("float32")
target = np.random.uniform(0, 1, (1, 5)).astype("float32")
loss = criterion.forward(input, target)

> input
array([[ 0.34291017,  0.95894575,  0.23869193,  0.42518589,  0.73902631]], dtype=float32)

> target 
array([[ 0.00489056,  0.7253111 ,  0.94344038,  0.69811821,  0.45532107]], dtype=float32)

> loss
0.20651573

```
