## VolumetricMaxPooling ##

**Scala:**
```scala
val layer = VolumetricMaxPooling[T](
  kernelT, kernelW, kernelH,
  strideT, strideW, strideH,
  paddingT, paddingW, paddingH
)
```
**Python:**
```python
layer = VolumetricMaxPooling(
  kernelT, kernelW, kernelH,
  strideT, strideW, strideH,
  paddingT, paddingW, paddingH
)
```

Applies 3D max-pooling operation in kT x kW x kH regions by step size dT x dW x dH.
The number of output features is equal to the number of input planes / dT.
The input can optionally be padded with zeros. Padding should be smaller than
half of kernel size. That is, padT < kT/2, padW < kW/2 and padH < kH/2

The input layout should be [batch, plane, time, height, width] or [plane, time, height, width]

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor

val layer = VolumetricMaxPooling[Float](
  2, 2, 2,
  1, 1, 1,
  0, 0, 0
)

val input = Tensor[Float](T(T(
  T(
    T(1.0f, 2.0f, 3.0f),
    T(4.0f, 5.0f, 6.0f),
    T(7.0f, 8.0f, 9.0f)
  ),
  T(
    T(4.0f, 5.0f, 6.0f),
    T(1.0f, 3.0f, 9.0f),
    T(2.0f, 3.0f, 7.0f)
  )
)))
layer.forward(input)
layer.backward(input, Tensor[Float](T(T(T(
  T(0.1f, 0.2f),
  T(0.3f, 0.4f)
)))))
```

Its output should be
```
(1,1,.,.) =
5.0     9.0
8.0     9.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x2x2]

(1,1,.,.) =
0.0     0.0     0.0
0.0     0.1     0.0
0.0     0.3     0.4

(1,2,.,.) =
0.0     0.0     0.0
0.0     0.0     0.2
0.0     0.0     0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x3x3]
```

**Python example:**
```python
from bigdl.nn.layer import VolumetricMaxPooling
import numpy as np

layer = VolumetricMaxPooling(
  2, 2, 2,
  1, 1, 1,
  0, 0, 0
)

input = np.array([[
  [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
  ],
  [
    [4.0, 5.0, 6.0],
    [1.0, 3.0, 9.0],
    [2.0, 3.0, 7.0]
  ]
]])
layer.forward(input)
layer.backward(input, np.array([[[
  [0.1, 0.2],
  [0.3, 0.4]
]]]))
```


Its output should be
```
array([[[[ 5.,  9.],
         [ 8.,  9.]]]], dtype=float32)

array([[[[ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.1       ,  0.        ],
         [ 0.        ,  0.30000001,  0.40000001]],

        [[ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.2       ],
         [ 0.        ,  0.        ,  0.        ]]]], dtype=float32)
```