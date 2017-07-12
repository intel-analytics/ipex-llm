## SpatialAveragePooling ##

**Scala:**
```scala
val m = SpatialAveragePooling(kW, kH, dW=1, dH=1, padW=0, padH=0, ceilMode=false, countIncludePad=true, divide=true)
```
**Python:**
```python
m = SpatialAveragePooling(kw, kh, dw=1, dh=1, pad_w=0, pad_h=0,ceil_mode=False, count_include_pad=True, divide=True)
```

SpatialAveragePooling is a module that applies 2D average-pooling operation in `kW`x`kH` regions by step size `dW`x`dH`.

The number of output features is equal to the number of input planes.

**Scala example:**
```scala
scala> 
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val input = Tensor(1, 3, 3).randn()
val m = SpatialAveragePooling(3, 2, 2, 1)
val output = m.forward(input)
val gradOut = Tensor(1, 2, 1).randn()
val gradIn = m.backward(input,gradOut)

scala> print(input)
(1,.,.) =
0.9916249       1.0299556       0.5608558
-0.1664829      1.5031902       0.48598626
0.37362042      -0.0966136      -1.4257964

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x3x3]

scala> print(output)
(1,.,.) =
0.7341883
0.1123173

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x1]

scala> print(gradOut)
(1,.,.) =
-0.42837557
-1.5104272

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x2x1]

scala> print(gradIn)
(1,.,.) =
-0.071395926    -0.071395926    -0.071395926
-0.3231338      -0.3231338      -0.3231338
-0.25173786     -0.25173786     -0.25173786

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x3]


```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input = np.random.randn(1,3,3)
print "input is :",input

m = SpatialAveragePooling(3,2,2,1)
out = m.forward(input)
print "output of m is :",out

grad_out = np.random.rand(1,3,1)
grad_in = m.backward(input,grad_out)
print "grad input of m is :",grad_in
```
produces output:

```python
input is : [[[ 1.50602425 -0.92869054 -1.9393117 ]
  [ 0.31447547  0.63450578 -0.92485516]
  [-2.07858315 -0.05688643  0.73648798]]]
creating: createSpatialAveragePooling
output of m is : [array([[[-0.22297533],
        [-0.22914261]]], dtype=float32)]
grad input of m is : [array([[[ 0.06282618,  0.06282618,  0.06282618],
        [ 0.09333335,  0.09333335,  0.09333335],
        [ 0.03050717,  0.03050717,  0.03050717]]], dtype=float32)]

```