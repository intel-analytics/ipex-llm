## SpatialAveragePooling ##

**Scala:**
```scala
val m = SpatialAveragePooling[T](kW, kH, dW, dH, padW, padH, ceilMode, countIncludePad, divide)
```
**Python:**
```python
m = SpatialAveragePooling(kw, kh, dw=1, dh=1, pad_w=0, pad_h=0,ceil_mode=False, count_include_pad=True, divide=True)
```

SpatialAveragePooling is a module that applies 2D average-pooling operation in kWxkH regions by step size dWxdH.

The number of output features is equal to the number of input planes.

**Scala example:**
```scala
scala> val input = Tensor[Double](1, 3, 3).randn()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,.,.) =
-0.432000902270295      0.028882976042042544    1.066722105416054
1.5154048822103217      2.6621710182077405      -0.33727350742325946
0.04048038473302521     -0.5231959373696738     1.952579854364095

[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 1x3x3]

scala> val m = SpatialAveragePooling[Double](3, 2, 2, 1)
m: com.intel.analytics.bigdl.nn.SpatialAveragePooling[Double] = SpatialAveragePooling$mcD$sp[27778a9e](3, 2, 2, 1, 0, 0)

scala> m.forward(input)
res12: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,.,.) =
0.7506510953637674
0.8850277824537081

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x1]

scala> val gradOutput = Tensor[Double](1, 2, 1).randn()
gradOutput: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,.,.) =
-0.26737872039319
-0.1198219665838401

[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 1x2x1]

scala> m.backward(input,gradOut)
res13: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,.,.) =
0.016666666666666666    0.016666666666666666    0.016666666666666666
0.03333333333333333     0.03333333333333333     0.03333333333333333
0.016666666666666666    0.016666666666666666    0.016666666666666666

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