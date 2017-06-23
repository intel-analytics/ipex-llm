## DistKLDivCriterion ##

**Scala:**
```scala
val loss = DistKLDivCriterion[T](sizeAverage)
```
**Python:**
```python
loss = DistKLDivCriterion(size_average=True)
```

DistKLDivCriterion is the Kullback–Leibler divergence loss.

**Scala example:**
```scala
scala> val input = Tensor[Double](2).randn()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
1.2477045608450934
-0.481583389657839
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2]

scala> val target = Tensor[Double](2).randn()
target: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.4163043798676898
-0.4280601053062339
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2]

scala> val loss = DistKLDivCriterion[Double]()
loss: com.intel.analytics.bigdl.nn.DistKLDivCriterion[Double] = nn.DistKLDivCriterion (true)

scala> loss.forward(input,target)
res30: Double = -0.44212423625480407

scala> loss.backward(input,target)
res31: com.intel.analytics.bigdl.tensor.Tensor[Double] =
-0.2081521899338449
0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
```

**Python example:**
```python
from bigdl.nn.criterion import *
import numpy as np

input  = np.random.randn(2)
target = np.array([2,1],dtype='float64')

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
input= [ 0.89250893  0.70999532]
target= [ 2.  1.]
creating: createDistKLDivCriterion
output of loss is : -0.55435944
grad out of loss is : [-1.  -0.5]
```
