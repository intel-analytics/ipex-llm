## SpatialMaxPooling ##

**Scala:**
```scala
val mp = SpatialMaxPooling(2, 2, 2, 2)
```
**Python:**
```python
mp = SpatialMaxPooling(2, 2, 2, 2)
```

Applies 2D max-pooling operation in kWxkH regions by step size dWxdH steps.
The number of output features is equal to the number of input planes.
If the input image is a 3D tensor nInputPlane x height x width,
the output image size will be nOutputPlane x oheight x owidth where
owidth  = op((width  + 2*padW - kW) / dW + 1)
oheight = op((height + 2*padH - kH) / dH + 1)
op is a rounding operator. By default, it is floor.
It can be changed by calling :ceil() or :floor() methods.

**Scala example:**

```scala
import com.intel.analytics.bigdl.nn.SpatialMaxPooling
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val mp = SpatialMaxPooling(2, 2, 2, 2)
val input = Tensor(1, 3, 3)

input(Array(1, 1, 1)) = 0.5336726f
input(Array(1, 1, 2)) = 0.7963769f
input(Array(1, 1, 3)) = 0.5674766f
input(Array(1, 2, 1)) = 0.1803996f
input(Array(1, 2, 2)) = 0.2460861f
input(Array(1, 2, 3)) = 0.2295625f
input(Array(1, 3, 1)) = 0.3073633f
input(Array(1, 3, 2)) = 0.5973460f
input(Array(1, 3, 3)) = 0.4298954f

val gradOutput = Tensor(1, 1, 1)
gradOutput(Array(1, 1, 1)) = 0.023921491578221f

val output = mp.forward(input)
val gradInput = mp.backward(input, gradOutput)

println(output)
println(gradInput)
```

The output is,

```
(1,.,.) =
0.7963769

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x1]
```

The gradInput is,

```
(1,.,.) =
0.0     0.023921492     0.0
0.0     0.0     0.0
0.0     0.0     0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x3]
```

**Python example:**

```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

mp = SpatialMaxPooling(2, 2, 2, 2)


input = np.array([0.5336726, 0.7963769, 0.5674766, 0.1803996, 0.2460861, 0.2295625, 0.3073633, 0.5973460, 0.4298954]).astype("float32")
input = input.reshape(1, 3, 3)

output = mp.forward(input)
print output

gradOutput = np.array([0.023921491578221]).astype("float32")
gradOutput = gradOutput.reshape(1, 1, 1)

gradInput = mp.backward(input, gradOutput)
print gradInput
```

The output is,

```
[array([[[ 0.79637688]]], dtype=float32)]
```

The gradInput is,

```
[array([[[ 0.        ,  0.02392149,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]]], dtype=float32)]
```