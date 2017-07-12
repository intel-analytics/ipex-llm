## SpatialDilatedConvolution ##

**Scala:**
```scala
val layer = SpatialDilatedConvolution[T](
  inputPlanes,
  outputPlanes,
  kernelW,
  kernelH,
  strideW,
  strideH,
  paddingW,
  paddingH,
  dilationW,
  dilationH
)
```
**Python:**
```python
layer = SpatialDilatedConvolution(
  inputPlanes,
  outputPlanes,
  kernelW,
  kernelH,
  strideW,
  strideH,
  paddingW,
  paddingH,
  dilationW,
  dilationH
)
```

Apply a 2D dilated convolution over an input image.

The input tensor is expected to be a 3D or 4D(with batch) tensor.

For a normal SpatialConvolution, the kernel will multiply with input
image element-by-element contiguous. In dilated convolution, it’s possible
to have filters that have spaces between each cell. For example, filter w and
image x, when dilatiionW and dilationH both = 1, this is normal 2D convolution
```
w(0, 0) * x(0, 0), w(0, 1) * x(0, 1)
w(1, 0) * x(1, 0), w(1, 1) * x(1, 1)
```
when dilationW and dilationH both = 2
```
w(0, 0) * x(0, 0), w(0, 1) * x(0, 2)
w(1, 0) * x(2, 0), w(1, 1) * x(2, 2)
```
when dilationW and dilationH both = 3
```
w(0, 0) * x(0, 0), w(0, 1) * x(0, 3)
w(1, 0) * x(3, 0), w(1, 1) * x(3, 3)
```

If input is a 3D tensor nInputPlane x height x width,
 * owidth  = floor(width + 2 * padW - dilationW * (kW-1) - 1) / dW + 1
 * oheight = floor(height + 2 * padH - dilationH * (kH-1) - 1) / dH + 1

Reference Paper:
> Yu F, Koltun V. Multi-scale context aggregation by dilated convolutions[J].
arXiv preprint arXiv:1511.07122, 2015.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor

val layer = SpatialDilatedConvolution[Float](1, 1, 2, 2, 1, 1, 0, 0, 2, 2)
val input = Tensor[Float](T(T(
  T(1.0f, 2.0f, 3.0f, 4.0f),
  T(5.0f, 6.0f, 7.0f, 8.0f),
  T(9.0f, 1.0f, 2.0f, 3.0f),
  T(4.0f, 5.0f, 6.0f, 7.0f)
)))
val filter = Tensor[Float](T(T(T(
  T(1.0f, 1.0f),
  T(1.0f, 1.0f)
))))
layer.weight.copy(filter)
layer.bias.zero()
layer.forward(input)
layer.backward(input, Tensor[Float](T(T(
  T(0.1f, 0.2f),
  T(0.3f, 0.4f)
))))
```

Its output should be
```
(1,.,.) =
15.0    10.0
22.0    26.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x2]

(1,.,.) =
0.1     0.2     0.1     0.2
0.3     0.4     0.3     0.4
0.1     0.2     0.1     0.2
0.3     0.4     0.3     0.4

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x4x4]
```

**Python example:**
```python
from bigdl.nn.layer import SpatialDilatedConvolution
import numpy as np

layer = SpatialDilatedConvolution(1, 1, 2, 2, 1, 1, 0, 0, 2, 2)
input = np.array([[
  [1.0, 2.0, 3.0, 4.0],
  [5.0, 6.0, 7.0, 8.0],
  [9.0, 1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0, 7.0]
]])
filter = np.array([[[
  [1.0, 1.0],
  [1.0, 1.0]
]]])
bias = np.array([0.0])
layer.set_weights([filter, bias])
layer.forward(input)
layer.backward(input, np.array([[[0.1, 0.2], [0.3, 0.4]]]))
```

Its output should be
```
array([[[ 15.,  10.],
        [ 22.,  26.]]], dtype=float32)
        
array([[[ 0.1       ,  0.2       ,  0.1       ,  0.2       ],
        [ 0.30000001,  0.40000001,  0.30000001,  0.40000001],
        [ 0.1       ,  0.2       ,  0.1       ,  0.2       ],
        [ 0.30000001,  0.40000001,  0.30000001,  0.40000001]]], dtype=float32)

```