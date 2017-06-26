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

> print(softSign.forward(Tensor(3, 3).rand()))
0.45222705028881033	0.1911821164520032	0.3135549602242586	
0.43199253473543525	0.4627414232312455	0.12919941223249864	
0.20673001813061254	0.15548499588691123	0.3566450036430123	
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
