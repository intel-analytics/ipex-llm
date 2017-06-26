## Exp ##

**Scala:**
```scala
val module = Exp()
```
**Python:**
```python
module = Exp()
```

Exp applies element-wise exp operation to input tensor


**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val exp = Exp()

> print(exp.forward(Tensor(3, 3).rand()))
2.117167441009217	1.3348497682757767	2.597868000359312	
2.3517183035087625	1.2622098046468193	1.3445996186474545	
1.4186639561465524	2.1381977568275885	1.866953124359979	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

```

**Python example:**
```python
from bigdl.nn.layer import *
exp = Exp()
> exp.forward(np.array([[1, 2, 3],[1, 2, 3]]))
[array([[  2.71828175,   7.38905621,  20.08553696],
       [  2.71828175,   7.38905621,  20.08553696]], dtype=float32)]

```
