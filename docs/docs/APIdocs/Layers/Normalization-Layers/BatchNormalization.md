## BatchNormalization ##

**Scala:**
```scala
val bn = BatchNormalization(nOutput, eps, momentum, affine)
```
**Python:**
```python
bn = BatchNormalization(n_output, eps, momentum, affine)
```

This layer implements Batch Normalization as described in the paper:
[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
by Sergey Ioffe, Christian Szegedy

This implementation is useful for inputs NOT coming from convolution layers. For convolution layers, use nn.SpatialBatchNormalization.

The operation implemented is:

```
              ( x - mean(x) )
      y =  -------------------- * gamma + beta
              standard-deviation(x)
```
where gamma and beta are learnable parameters.The learning of gamma and beta is optional.

**Parameters:**
* **nOutput** - feature map number
* **eps** - avoid divide zero. Default: 1e-5
* **momentum** - momentum for weight update. Default: 0.1
* **affine** - affine operation on output or not. Default: true

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val bn = BatchNormalization(2)
val input = Tensor(T(
             T(1.0f, 2.0f),
             T(3.0f, 6.0f))
            )
val gradOutput = Tensor(T(
             T(1.0f, 2.0f),
             T(3.0f, 6.0f))
)
val output = bn.forward(input)
val gradient = bn.backward(input, gradOutput)
-> print(output) 
# There's random factor. An output could be
-0.46433213     -0.2762179      
0.46433213      0.2762179       
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
-> print(gradient)
# There's random factor. An output could be
-4.649627E-6    -6.585548E-7    
4.649627E-6     6.585548E-7     
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
bn = BatchNormalization(2)
input = np.array([
  [1.0, 2.0],
  [3.0, 6.0]
])
grad_output = np.array([
           [2.0, 3.0],
           [4.0, 5.0]
         ])
output = bn.forward(input)
gradient = bn.backward(input, grad_output)
-> print output
# There's random factor. An output could be
[[-0.99583918 -0.13030811]
 [ 0.99583918  0.13030811]]
-> print gradient
# There's random factor. An output could be
[[ -9.97191637e-06  -1.55339364e-07]
 [  9.97191637e-06   1.55339364e-07]]
```
