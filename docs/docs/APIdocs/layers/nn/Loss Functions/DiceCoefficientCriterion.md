## DiceCoefficientCriterion ##

**Scala:**
```scala
val loss = DiceCoefficientCriterion[T](sizeAverage, epsilon)
```
**Python:**
```python
loss = DiceCoefficientCriterion(size_average,epsilon)
```

DiceCoefficientCriterion is the Dice-Coefficient objective function. 

Both `forward` and `backward` accept two tensors : input and target. The `forward` result is formulated as 
          `1 - (2 * (input intersection target) / (input union target))`

**Scala example:**
```scala
scala> val input = Tensor[Double](2).randn()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
-1.6529486997146012
-0.16373539545619223
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2]

scala> val target = Tensor[Double](Storage[Double](Array(2.0d, 1.0d)))
target: com.intel.analytics.bigdl.tensor.Tensor[Double] =
2.0
1.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2]

scala> val loss = DiceCoefficientCriterion[Double](epsilon = 1.0f)
loss: com.intel.analytics.bigdl.nn.DiceCoefficientCriterion[Double] = nn.DiceCoefficientCriterion

scala> loss.forward(input,target)
res28: Double = 3.7202960307456734

scala> loss.backward(input,target)
res29: com.intel.analytics.bigdl.tensor.Tensor[Double] =
-3.0780227524020987     -2.16198490575963
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2]
```

**Python example:**
```python
from bigdl.nn.criterion import *
import numpy as np

input  = np.random.randn(2)
target = np.array([2,1],dtype='float64')

print "input=", input
print "target=", target
loss = DiceCoefficientCriterion(size_average=True,epsilon=1.0)
out = loss.forward(input,target)
print "output of loss is :",out

grad_out = loss.backward(input,target)
print "grad out of loss is :",grad_out
```
produces output:
```python
input= [ 0.4440505  2.9430301]
target= [ 2.  1.]
creating: createDiceCoefficientCriterion
output of loss is : -0.17262316
grad out of loss is : [[-0.38274616 -0.11200322]]
```