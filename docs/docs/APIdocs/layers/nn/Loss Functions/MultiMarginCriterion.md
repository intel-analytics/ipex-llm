## MultiMarginCriterion ##

**Scala:**
```scala
val loss = MultiMarginCriterion[T](p,weights,margin,sizeAverage)
```
**Python:**
```python
loss = MultiMarginCriterion(p=1,weights=None,margin=1.0,size_average=True)
```

MultiMarginCriterion is a loss function that optimizes a multi-class classification hinge loss (margin-based loss) between input x and output y (y is the target class index).

**Scala example:**
```scala
scala> val input = Tensor[Double](3, 2).randn()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.27774379458799353     -0.39997834149200373
-0.029815703456747233   -0.44182467832945155
-1.4002233278913043     0.898932679101772
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x2]

scala> val target = Tensor[Double](Storage[Double](Array(2.0d, 1.0d,2.0d)))
target: com.intel.analytics.bigdl.tensor.Tensor[Double] =
2.0
1.0
2.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]

scala> val loss =  MultiMarginCriterion[Double](1)
loss: com.intel.analytics.bigdl.nn.MultiMarginCriterion[Double] = nn.MultiMarginCriterion(true, null, 1.0)

scala> loss.forward(input,target)
res21: Double = 0.37761886020121543

scala> loss.backward(input,target)
res22: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.16666666666666666     -0.16666666666666666
-0.16666666666666666    0.16666666666666666
0.0     0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2]

```

**Python example:**
```python
from bigdl.nn.criterion import *
import numpy as np

input  = np.random.randn(3,2)
target = np.array([2,1,2],dtype='float64')

loss = MultiMarginCriterion(1)
out = loss.forward(input, target)
print "output of loss is :",out

grad_out = loss.backward(input,target)
print "grad out of loss is :",grad_out
```
produces output
```
creating: createMultiMarginCriterion
output of loss is : 0.24410136
grad out of loss is : [array([[ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.16666667, -0.16666667]], dtype=float32)]

```
