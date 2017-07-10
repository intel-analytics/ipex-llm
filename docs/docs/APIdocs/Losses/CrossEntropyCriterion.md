## CrossEntropyCriterion ##

**Scala:**
```scala
val module = CrossEntropyCriterion(weights, sizeAverage)
```
**Python:**
```python
module = CrossEntropyCriterion(weights, sizeAverage)
```

This criterion combines LogSoftMax and ClassNLLCriterion in one single class.

`weights` A tensor assigning weight to each of the classes

`sizeAverage` whether to divide the sequence length. Default is true.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.Storage

val layer = CrossEntropyCriterion[Double]()
val input = Tensor[Double](Storage(Array(
    1.0262627674932,
    -1.2412600935171,
    -1.0423174168648,
    -0.90330565804228,
    -1.3686840144413,
    -1.0778380454479,
    -0.99131220658219,
    -1.0559142847536,
    -1.2692712660404
    ))).resize(3, 3)
val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3
> print(layer.forward(input, target))
0.9483051199107635
```

**Python example:**
```python
from bigdl.nn.criterion import *

layer = CrossEntropyCriterion()
input = np.array([1.0262627674932,
                      -1.2412600935171,
                      -1.0423174168648,
                      -0.90330565804228,
                      -1.3686840144413,
                      -1.0778380454479,
                      -0.99131220658219,
                      -1.0559142847536,
                      -1.2692712660404
                      ]).reshape(3,3)
target = np.array([1, 2, 3])                      
>layer.forward(input, target)
0.94830513
```