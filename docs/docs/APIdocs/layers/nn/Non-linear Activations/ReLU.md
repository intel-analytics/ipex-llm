## ReLU ##

**Scala:**
```scala
val relu = ReLU(ip = false)
```
**Python:**
```python
relu = ReLU(ip)
```

ReLU applies the element-wise rectified linear unit (ReLU) function to the input

`ip` illustrate if the ReLU fuction is done on the origin input
```
ReLU function : `f(x) = max(0, x)`
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val relu = ReLU(false)

val input = Tensor(3, 3).rand()
> print(input)
0.13486342	0.8986828	0.2648762	
0.56467545	0.7727274	0.65959305	
0.01554346	0.9552375	0.2434533	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]

> print(relu.forward(input))
0.13486342	0.8986828	0.2648762	
0.56467545	0.7727274	0.65959305	
0.01554346	0.9552375	0.2434533	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]


```

**Python example:**
```python
from bigdl.nn.layer import *
relu = ReLU(False)
> relu.forward(np.array([[-1, -2, -3], [0, 0, 0], [1, 2, 3]]))
[array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.],
       [ 1.,  2.,  3.]], dtype=float32)]
     
```

