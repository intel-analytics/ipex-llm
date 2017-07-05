## DistKLDivCriterion ##

**Scala:**
```scala
val loss = DistKLDivCriterion[T](sizeAverage=true)
```
**Python:**
```python
loss = DistKLDivCriterion(size_average=True)
```

DistKLDivCriterion is the Kullback–Leibler divergence loss.

**Scala example:**
```scala

scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.Storage

val input = Tensor(2).randn()
val target = Tensor(Storage(Array(2.0f, 1.0f)))
val loss = DistKLDivCriterion()
val output = loss.forward(input,target)
val grad = loss.backward(input,target)

scala> print(input)
-0.3854126
-0.7707398
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2]

scala> print(target)
2.0
1.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2]

scala> print(output)
1.4639297

scala> print(grad)
-1.0
-0.5
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2]

```

**Python example:**
```python
from bigdl.nn.criterion import *
import numpy as np

input  = np.random.randn(2)
target = np.array([2,1])

print "input=", input
print "target=", target
loss = DistKLDivCriterion()
out = loss.forward(input,target)
print "output of loss is :",out

grad_out = loss.backward(input,target)
print "grad out of loss is :",grad_out
```
produces output:
```python
input= [-1.14333924  0.97662296]
target= [2 1]
creating: createDistKLDivCriterion
output of loss is : 1.348175
grad out of loss is : [-1.  -0.5]
```
