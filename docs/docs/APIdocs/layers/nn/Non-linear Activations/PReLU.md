## PReLU ##

**Scala:**
```scala
val module = PReLU(nOutputPlane: Int = 0)
```
**Python:**
```python
module = PReLU(nOutputPlane)
```

Applies parametric ReLU, which parameter varies the slope of the negative part.

```
PReLU: f(x) = max(0, x) + a * min(0, x)
```
nOutputPlane's default value is 0, that means using PReLU in shared version and has
only one parameters. nOutputPlane is the input map number(Default is 0).

Notice: Please don't use weight decay on this.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val module = PReLU(2)
val input = Tensor(2, 2, 3).randn()
val output = module.forward(input)

> input
(1,.,.) =
-0.17810068	-0.69607687	0.25582042
-1.2140307	-1.5410945	1.0209005

(2,.,.) =
0.2826971	0.6370953	0.21471702
-0.16203058	-0.5643519	0.816576

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x3]

> output
(1,.,.) =
-0.04452517	-0.17401922	0.25582042
-0.3035077	-0.38527364	1.0209005

(2,.,.) =
0.2826971	0.6370953	0.21471702
-0.040507644	-0.14108798	0.816576

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = PReLU(2)
input = np.random.randn(2, 2, 3)
output = module.forward(input)

> input
[[[ 2.50596953 -0.06593339 -1.90273409]
  [ 0.2464341   0.45941315 -0.41977094]]

 [[-0.8584367   2.19389229  0.93136755]
  [-0.39209027  0.16507514 -0.35850447]]]
  
> output
[array([[[ 2.50596952, -0.01648335, -0.47568351],
         [ 0.24643411,  0.45941314, -0.10494273]],
 
        [[-0.21460918,  2.19389224,  0.93136758],
         [-0.09802257,  0.16507514, -0.08962612]]], dtype=float32)]
```
