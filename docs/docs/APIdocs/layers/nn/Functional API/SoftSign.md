## SoftSign ##

**Scala:**
```scala
val softSign = SoftSign()
```
**Python:**
```python
softSign = SoftSign()
```

SoftSign applies SoftSign function to the input tensor

SoftSign function: `f_i(x) = x_i / (1+|x_i|)`


**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val softSign = SoftSign()
val input = Tensor(3, 3).rand()

> print(input)
0.6733504	0.7566517	0.43793806	
0.09683273	0.05829774	0.4567967	
0.20021072	0.11158377	0.31668025

> print(softSign.forward(input))
0.40239656	0.4307352	0.30455974	
0.08828395	0.05508633	0.31356242	
0.16681297	0.10038269	0.24051417	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]


```

**Python example:**
```python
from bigdl.nn.layer import *
softSign=SoftSign()
> softSign.forward(np.array([[1, 2, 4],[-1, -2, -4]]))
[array([[ 0.5       ,  0.66666669,  0.80000001],
       [-0.5       , -0.66666669, -0.80000001]], dtype=float32)]

```
