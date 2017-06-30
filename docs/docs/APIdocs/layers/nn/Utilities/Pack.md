## Pack ##

**Scala:**
```scala
val module = Pack(dim)
```
**Python:**
```python
module = Pack(dim)
```

Pack is used to stack a list of n-dimensional tensors into one (n+1)-dimensional tensor.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val module = Pack(2)
val input1 = Tensor(2, 2).randn()
val input2 = Tensor(2, 2).randn()
val input = T()
input(1) = input1
input(2) = input2

val output = module.forward(input)

> input
 {
	2: -0.8737048	-0.7337217
	   0.7268678	-0.53470045
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
	1: -1.3062215	-0.58756566
	   0.8921608	-1.8087773
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
 }

 
> output
(1,.,.) =
-1.3062215	-0.58756566
-0.8737048	-0.7337217

(2,.,.) =
0.8921608	-1.8087773
0.7268678	-0.53470045

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Pack(2)
input1 = np.random.randn(2, 2)
input2 = np.random.randn(2, 2)
input = [input1, input2]
output = module.forward(input)

> input
[array([[ 0.92741416, -3.29826586],
       [-0.03147819, -0.10049306]]), array([[-0.27146461, -0.25729802],
       [ 0.1316149 ,  1.27620145]])]
       
> output
array([[[ 0.92741418, -3.29826593],
        [-0.27146462, -0.25729802]],

       [[-0.03147819, -0.10049306],
        [ 0.13161489,  1.27620149]]], dtype=float32)
```
