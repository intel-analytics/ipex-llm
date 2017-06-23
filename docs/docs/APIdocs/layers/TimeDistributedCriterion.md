## TimeDistributedCriterion ##

**Scala:**
```scala
val module = TimeDistributedCriterion(critrn, sizeAverage)
```
**Python:**
```python
module = TimeDistributedCriterion(critrn, sizeAverage)
```

This class is intended to support inputs with 3 or more dimensions.
Apply Any Provided Criterion to every temporal slice of an input.
  
`critrn` embedded criterion

`sizeAverage` whether to divide the sequence length. Default is false.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.Storage

val criterion = ClassNLLCriterion[Double]()
val layer = TimeDistributedCriterion[Double](criterion, true)
val input = Tensor[Double](Storage(Array(
    1.0262627674932,
    -1.2412600935171,
    -1.0423174168648,
    -1.0262627674932,
    -1.2412600935171,
    -1.0423174168648,
    -0.90330565804228,
    -1.3686840144413,
    -1.0778380454479,
    -0.90330565804228,
    -1.3686840144413,
    -1.0778380454479,
    -0.99131220658219,
    -1.0559142847536,
    -1.2692712660404,
    -0.99131220658219,
    -1.0559142847536,
    -1.2692712660404))).resize(3, 2, 3)
val target = Tensor[Double](3, 2)
    target(Array(1, 1)) = 1
    target(Array(1, 2)) = 1
    target(Array(2, 1)) = 2
    target(Array(2, 2)) = 2
    target(Array(3, 1)) = 3
    target(Array(3, 2)) = 3
> print(layer.forward(input, target))
0.8793184268272332
```

**Python example:**
```python
from bigdl.nn.criterion import *

criterion = ClassNLLCriterion()
layer = TimeDistributedCriterion(criterion, True)
input = np.array([1.0262627674932,
                      -1.2412600935171,
                      -1.0423174168648,
                      -1.0262627674932,
                      -1.2412600935171,
                      -1.0423174168648,
                      -0.90330565804228,
                      -1.3686840144413,
                      -1.0778380454479,
                      -0.90330565804228,
                      -1.3686840144413,
                      -1.0778380454479,
                      -0.99131220658219,
                      -1.0559142847536,
                      -1.2692712660404,
                      -0.99131220658219,
                      -1.0559142847536,
                      -1.2692712660404]).reshape(3,2,3)
target = np.array([[1,1],[2,2],[3,3]])                      
>layer.forward(input, target)
0.8793184
```