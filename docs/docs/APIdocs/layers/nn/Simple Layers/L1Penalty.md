## L1Penalty ##

**Scala:**
```scala
val l1Penalty = L1Penalty(val l1weight: Int, val sizeAverage: Boolean = false,val provideOutput: Boolean = true)
```
**Python:**
```python
l1Penalty = L1Penalty( l1weight, size_average=False, provide_output=True)
```
L1Penalty adds an L1 penalty to an input 
For forward, the output is the same as input and a L1 loss of the latent state will be calculated each time
For backward, gradInput = gradOutput + gradLoss


**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val l1Penalty = L1Penalty(1, true, true)
val input = Tensor(3, 3).rand()

> print(input)
0.0370419	0.03080979	0.22083037	
0.1547358	0.018475588	0.8102709	
0.86393493	0.7081842	0.13717912	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]


> print(l1Penalty.forward(input))
0.0370419	0.03080979	0.22083037	
0.1547358	0.018475588	0.8102709	
0.86393493	0.7081842	0.13717912	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]	

```

**Python example:**
```python
from bigdl.nn.layer import *
l1Penalty = L1Penalty(1, True, True)

>>> l1Penalty.forward(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
[array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.],
       [ 7.,  8.,  9.]], dtype=float32)]

```

