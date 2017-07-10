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
val input = Tensor(3, 3).rand()

> print(input)
0.7056571	0.25239098	0.75746965	
0.89736927	0.31193605	0.23842576	
0.69492024	0.7512544	0.8386124	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]

> print(tanhShrink.forward(input))
0.09771085	0.0052260756	0.11788553	
0.18235475	0.009738684	0.004417494	
0.09378672	0.1153577	0.153539	
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

