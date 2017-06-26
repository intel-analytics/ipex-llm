## TanhShrink ##

**Scala:**
```scala
val tanhShrink = TanhShrink()
```
**Python:**
```python
tanhShrink = TanhShrink()
```
TanhShrink applies element-wise Tanh and Shrink function to the input

TanhShrink function : `f(x) = scala.math.tanh(x) - 1`

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val tanhShrink = TanhShrink()

>  print(tanhShrink.forward(Tensor(3, 3).rand()))
0.09803492	0.0072663724	0.12395775	
1.1608005E-5	0.123642504	0.039898574	
0.0033952743	0.015322447	0.12457663	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

```

**Python example:**
```python
from bigdl.nn.layer import *
tanhShrink = TanhShrink()

>  tanhShrink.forward(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
[array([[ 0.23840582,  1.03597236,  2.00494528],
       [ 3.00067067,  4.0000906 ,  5.0000124 ],
       [ 6.00000191,  7.        ,  8.        ]], dtype=float32)]

```

