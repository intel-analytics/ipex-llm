## ReLU ##

**Scala:**
```scala
val relu = ReLU(ip: Boolean = false)
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

val input = T()
input(1.0) = Tensor(1, 1).rand()
input(2.0) = Tensor(2, 2).rand()
input(3.0) = Tensor(3, 3).rand()

> print(relu.forward(Tensor(3, 3).rand()))
0.14756441	0.6177869	0.21481016	
0.98840165	0.95527816	0.17846434	
0.8655564	0.66041255	0.63329965	
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

