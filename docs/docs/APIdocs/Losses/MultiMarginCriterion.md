## MultiMarginCriterion ##

**Scala:**
```scala
val loss = MultiMarginCriterion(p=1,weights=null,margin=1.0,sizeAverage=true)
```
**Python:**
```python
loss = MultiMarginCriterion(p=1,weights=None,margin=1.0,size_average=True)
```

MultiMarginCriterion is a loss function that optimizes a multi-class classification hinge loss (margin-based loss) between input `x` and output `y` (`y` is the target class index).

**Scala example:**
```scala

scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.Storage

val input = Tensor(3,2).randn()
val target = Tensor(Storage(Array(2.0f, 1.0f, 2.0f)))
val loss = MultiMarginCriterion(1)
val output = loss.forward(input,target)
val grad = loss.backward(input,target)

scala> print(input)
-0.45896783     -0.80141246
0.22560088      -0.13517438
0.2601126       0.35492152
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]

scala> print(target)
2.0
1.0
2.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]

scala> print(output)
0.4811434

scala> print(grad)
0.16666667      -0.16666667
-0.16666667     0.16666667
0.16666667      -0.16666667
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2]


```

**Python example:**
```python
from bigdl.nn.criterion import *
import numpy as np

input  = np.random.randn(3,2)
target = np.array([2,1,2])
print "input=",input
print "target=",target

loss = MultiMarginCriterion(1)
out = loss.forward(input, target)
print "output of loss is : ",out

grad_out = loss.backward(input,target)
print "grad out of loss is : ",grad_out
```
produces output
```
input= [[ 0.46868305 -2.28562261]
 [ 0.8076243  -0.67809689]
 [-0.20342555 -0.66264743]]
target= [2 1 2]
creating: createMultiMarginCriterion
output of loss is :  0.8689213
grad out of loss is :  [[ 0.16666667 -0.16666667]
 [ 0.          0.        ]
 [ 0.16666667 -0.16666667]]


```
