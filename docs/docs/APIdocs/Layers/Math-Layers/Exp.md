## Exp ##

**Scala:**
```scala
val exp = Exp()
```
**Python:**
```python
exp = Exp()
```

Exp applies element-wise exp operation to input tensor


**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val exp = Exp()
val input = Tensor(3, 3).rand()
> print(input)
0.0858663	0.28117087	0.85724664	
0.62026995	0.29137492	0.07581586	
0.22099794	0.45131826	0.78286386	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]
> print(exp.forward(input))
1.0896606	1.32468		2.356663	
1.85943		1.3382663	1.078764	
1.2473209	1.5703809	2.1877286	
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
