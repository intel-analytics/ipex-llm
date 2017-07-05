## DiceCoefficientCriterion ##

**Scala:**
```scala
val loss = DiceCoefficientCriterion(sizeAverage=true, epsilon=1.0f)
```
**Python:**
```python
loss = DiceCoefficientCriterion(size_average=True,epsilon=1.0)
```

DiceCoefficientCriterion is the Dice-Coefficient objective function. 

Both `forward` and `backward` accept two tensors : input and target. The `forward` result is formulated as 
          `1 - (2 * (input intersection target) / (input union target))`

**Scala example:**
```scala

scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.Storage

val input = Tensor(2).randn()
val target = Tensor(Storage(Array(2.0f, 1.0f)))
val loss = DiceCoefficientCriterion(epsilon = 1.0f)
val output = loss.forward(input,target)
val grad = loss.backward(input,target)

scala> print(input)
-0.50278
0.51387966
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2]

scala> print(target)
2.0
1.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2]

scala> print(output)
0.9958517

scala> print(grad)
-0.99619853     -0.49758217
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