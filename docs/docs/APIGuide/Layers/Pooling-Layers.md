## SpatialMaxPooling ##

**Scala:**
```scala
val mp = SpatialMaxPooling(2, 2, dW=2, dH=2, padW=0, padH=0, format=DataFormat.NCHW)
```
**Python:**
```python
mp = SpatialMaxPooling(2, 2, dw=2, dh=2, pad_w=0, pad_h=0, to_ceil=false, format="NCHW")
```

Applies 2D max-pooling operation in kWxkH regions by step size dWxdH steps.
The number of output features is equal to the number of input planes.
If the input image is a 3D tensor nInputPlane x height x width,
the output image size will be nOutputPlane x oheight x owidth where

+ owidth  = op((width  + 2*padW - kW) / dW + 1)
+ oheight = op((height + 2*padH - kH) / dH + 1)

op is a rounding operator. By default, it is floor.
It can be changed by calling ceil() or floor() methods.

As for padding, when padW and padH are both -1, we use a padding algorithm similar to the "SAME" padding of tensorflow. That is
```scala
 outHeight = Math.ceil(inHeight.toFloat/strideH.toFloat)
 outWidth = Math.ceil(inWidth.toFloat/strideW.toFloat)

 padAlongHeight = Math.max(0, (outHeight - 1) * strideH + kernelH - inHeight)
 padAlongWidth = Math.max(0, (outWidth - 1) * strideW + kernelW - inWidth)

 padTop = padAlongHeight / 2
 padLeft = padAlongWidth / 2
```

The format parameter is a string value (or DataFormat Object in Scala) of "NHWC" or "NCHW" to specify the input data format of this layer. In "NHWC" format
data is stored in the order of \[batch_size, height, width, channels\], in "NCHW" format data is stored
in the order of \[batch_size, channels, height, width\].

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

## SpatialAveragePooling ##

**Scala:**
```scala
val m = SpatialAveragePooling(kW, kH, dW=1, dH=1, padW=0, padH=0, globalPooling=false, ceilMode=false, countIncludePad=true, divide=true, format=DataFormat.NCHW)
```
**Python:**
```python
m = SpatialAveragePooling(kw, kh, dw=1, dh=1, pad_w=0, pad_h=0, global_pooling=False, ceil_mode=False, count_include_pad=True, divide=True, format="NCHW")
```

SpatialAveragePooling is a module that applies 2D average-pooling operation in `kW`x`kH` regions by step size `dW`x`dH`.

The number of output features is equal to the number of input planes.

As for padding, when padW and padH are both -1, we use a padding algorithm similar to the "SAME" padding of tensorflow. That is
```scala
 outHeight = Math.ceil(inHeight.toFloat/strideH.toFloat)
 outWidth = Math.ceil(inWidth.toFloat/strideW.toFloat)

 padAlongHeight = Math.max(0, (outHeight - 1) * strideH + kernelH - inHeight)
 padAlongWidth = Math.max(0, (outWidth - 1) * strideW + kernelW - inWidth)

 padTop = padAlongHeight / 2
 padLeft = padAlongWidth / 2
```

The format parameter is a string value (or DataFormat Object in Scala) of "NHWC" or "NCHW" to specify the input data format of this layer. In "NHWC" format
data is stored in the order of \[batch_size, height, width, channels\], in "NCHW" format data is stored
in the order of \[batch_size, channels, height, width\].

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
## VolumetricMaxPooling ##

**Scala:**
```scala
val layer = VolumetricMaxPooling(
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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val layer = VolumetricMaxPooling(
  2, 2, 2,
  1, 1, 1,
  0, 0, 0
)

val input = Tensor(T(T(
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
layer.backward(input, Tensor(T(T(T(
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

## VolumetricAveragePooling ##

**Scala:**
```scala
val layer = VolumetricMaxPooling(
  kT, kW, kH, dT, dW, dH,
  padT=0, padW=0, padH=0,
  countIncludePad=true, ceilMode=false
)
```
**Python:**
```python
layer = VolumetricMaxPooling(
  k_t, k_w, k_h, d_t, d_w, d_h
  pad_t=0, pad_w=0, pad_h=0,
  count_include_pad=True, ceil_mode=False
)
```

Applies 3D average-pooling operation in kernel kT x kW x kH regions by step size dT x dW x dH.
The number of output features is equal to the number of input planes / dT.
The input can optionally be padded with zeros. Padding should be smaller than
half of kernel size. That is, padT < kT/2, padW < kW/2 and padH < kH/2

The input layout should be [batch, plane, time, height, width] or [plane, time, height, width]

By default, countIncludePad=true, which means to include padding when dividing the number of elements in pooling region.
One can use ceilMode to control whether the output size is to be ceiled or floored.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val layer = VolumetricAveragePooling(
  2, 2, 2,
  1, 1, 1,
  0, 0, 0
)

val input = Tensor(T(T(
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
layer.backward(input, Tensor(T(T(T(
  T(0.1f, 0.2f),
  T(0.3f, 0.4f)
)))))
```

Its output should be
```
(1,1,.,.) =
3.125	4.875
4.125	6.25

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x2x2]

(1,1,.,.) =
0.0125	0.0375	0.025
0.05	0.125	0.075
0.0375	0.087500006	0.05

(1,2,.,.) =
0.0125	0.0375	0.025
0.05	0.125	0.075
0.0375	0.087500006	0.05

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x3x3]
```

**Python example:**
```python
from bigdl.nn.layer import VolumetricAveragePooling
import numpy as np

layer = VolumetricAveragePooling(
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
array([[[[ 3.125  4.875]
         [ 4.125  6.25 ]]]], dtype=float32)

array([[[[ 0.0125      0.0375      0.025     ]
         [ 0.05        0.125       0.075     ]
         [ 0.0375      0.08750001  0.05      ]]

        [[ 0.0125      0.0375      0.025     ]
         [ 0.05        0.125       0.075     ]
         [ 0.0375      0.08750001  0.05      ]]]], dtype=float32)
```

## RoiPooling ##

**Scala:**
```scala
val m =  RoiPooling(pooled_w, pooled_h, spatial_scale)
```
**Python:**
```python
m = RoiPooling(pooled_w, pooled_h, spatial_scale)
```

RoiPooling is a module that performs Region of Interest pooling. 

It uses max pooling to convert the features inside any valid region of interest into a small feature map with a fixed spatial extent of pooledH × pooledW (e.g., 7 × 7).

An RoI is a rectangular window into a conv feature map. Each RoI is defined by a four-tuple (x1, y1, x2, y2) that specifies its top-left corner (x1, y1) and its bottom-right corner (x2, y2).

RoI max pooling works by dividing the h × w RoI window into an pooledH × pooledW grid of sub-windows of approximate size h/H × w/W and then max-pooling the values in each sub-window into the corresponding output grid cell. Pooling is applied independently to each feature map channel

`forward` accepts a table containing 2 tensors as input, the first tensor is the input image, the second tensor is the ROI regions. The dimension of the second tensor should be (*,5) (5 are  `batch_num, x1, y1, x2, y2`).  

**Scala example:**
```scala
scala> 
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.Storage
import com.intel.analytics.bigdl.utils.T

val input_data = Tensor(2,2,6,8).randn()
val rois = Array(0, 0, 0, 7, 5, 1, 6, 2, 7, 5, 1, 3, 1, 6, 4, 0, 3, 3, 3, 3)
val input_rois = Tensor(Storage(rois.map(x => x.toFloat))).resize(4, 5)
val input = T(input_data,input_rois)
val m = RoiPooling(3, 2, 1)
val output = m.forward(input)

scala> print(input)
 {
        2: 0.0  0.0     0.0     7.0     5.0
           1.0  6.0     2.0     7.0     5.0
           1.0  3.0     1.0     6.0     4.0
           0.0  3.0     3.0     3.0     3.0
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 4x5]
        1: (1,1,.,.) =
           0.48066297   1.0994664       0.32474303      2.3391871       -0.79605865     0.836963950.36107457      1.2622415
           0.657079     0.12720469      0.39894578      -0.41185552     -0.53111094     -0.36016005       -0.9726423      -2.5785272
           0.3091435    -0.03613516     0.2375721       -1.1920663      -0.6757661      1.10612681.5409279        -0.17411499
           0.23274016   -0.7149633      0.5473105       -0.40570387     -1.7966263      0.2071798-1.1530842       -0.010083453
           -1.5769979   0.17043112      -0.28578365     -0.90779626     0.61457515      -0.1553582-0.3912479      -0.15326484
           -0.24283029  1.3215472       1.3795123       -0.36933053     0.7077386       -0.56398267       0.6159163       0.5802894

           (1,2,.,.) =
           -1.1817129   -0.20470592     -1.3201113      0.36523122      -0.18260211     1.30210171.214403 1.1019816
           0.7186407    0.78731173      1.5452348       0.0396181       0.5927014       1.17697431.0501136        -0.58295316
           -0.96753055  0.6427254       -1.1396345      0.8701054       -0.22860864     -1.18719451.3372624       0.8616691
           0.796831     -0.16609778     0.2950535       0.4595303       0.192339        0.6086106-0.76351887      -0.65964174
           -0.12746814  -0.036058053    0.8858275       0.9677718       -1.1074747      -1.36859390.8783633       -0.11723315
           -0.6947403   -0.23226547     -1.8510057      -1.3695518      -0.22317407     -0.36249024       -0.24097045     1.5691053

           (2,1,.,.) =
           0.84056973   1.144949        -1.0660682      0.4416162       -0.94440234     -0.24461010.91145027      -0.88650835
           -0.81542057  0.14578317      -0.6531974      0.60776395      -0.32058007     -1.80771481.7660322       1.0680646
           1.1328241    0.43677545      -0.9402618      -1.3002211      0.26012567      1.69481340.37849447       0.39286092
           1.9443163    0.5415504       1.0793099       1.3312546       0.48346 1.2019655       0.3718734 0.21091922
           0.5499047    1.6418253       0.8064177       0.37626198      0.8736181       -0.40816033       -0.5806787      1.286581
           -0.5904657   -0.21188398     -0.040509004    1.2989452       1.6827602       1.3229258-0.68433124      0.87974

           (2,2,.,.) =
           -0.09759476  -0.32767114     0.16223079      2.3114302       -0.48496276     1.19290720.8572289        0.43429425
           -1.0245247   0.19002944      1.5659521       -1.3689835      -1.4437296      -0.38216656       0.6333655       -0.57124794
           -0.31111157  1.5184602       -1.3835855      -0.9295573      2.244521        -1.11849820.5451996       -0.4441631
           -1.534093    -0.5599659      1.1980947       -1.0140935      1.3288999       0.19487387-0.1261734      -1.2222558
           -0.070535585 0.9047848       -0.6719811      -1.6532638      -0.5290511      -0.18300447       0.69385433      0.018756092
           0.24767837   0.620484        -0.5346291      1.0685066       -0.36903372     -0.26955062       1.1042496       0.5944603

           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x6x8]
 }
 
scala> print(output)
(1,1,.,.) =
1.0994664       2.3391871       1.5409279
1.3795123       1.3795123       0.6159163

(1,2,.,.) =
1.5452348       1.5452348       1.3372624
0.8858275       0.9677718       1.5691053

(2,1,.,.) =
0.37849447      0.39286092      0.39286092
-0.5806787      1.286581        1.286581

(2,2,.,.) =
0.5451996       0.5451996       -0.4441631
1.1042496       1.1042496       0.5944603

(3,1,.,.) =
0.60776395      1.6948134       1.7660322
1.3312546       1.2019655       1.2019655

(3,2,.,.) =
2.244521        2.244521        0.6333655
1.3288999       1.3288999       0.69385433

(4,1,.,.) =
-0.40570387     -0.40570387     -0.40570387
-0.40570387     -0.40570387     -0.40570387

(4,2,.,.) =
0.4595303       0.4595303       0.4595303
0.4595303       0.4595303       0.4595303

[com.intel.analytics.bigdl.tensor.DenseTensor of size 4x2x2x3]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input_data = np.random.randn(2,2,6,8)
input_rois = np.array([0, 0, 0, 7, 5, 1, 6, 2, 7, 5, 1, 3, 1, 6, 4, 0, 3, 3, 3, 3],dtype='float64').reshape(4,5)
print "input is :",[input_data, input_rois]

m = RoiPooling(3,2,1.0)
out = m.forward([input_data,input_rois])
print "output of m is :",out
```
produces output:

```python
input is : [array([[[[ 0.08500103,  0.33421796,  0.29084699,  1.60344635, -0.24289341,
          -0.4793888 ,  0.09452426,  0.16842477],
         [-1.18575497, -0.53337542,  0.11661001,  0.9647904 , -0.25187936,
           0.36516823, -0.16647209, -0.08095158],
         [ 1.1982232 , -0.33549174,  0.11721347, -0.29319686, -0.01290122,
           0.12344296,  0.30074829, -2.34951463],
         [-0.60470899, -0.84657051,  0.1269276 , -0.06152321, -1.68838416,
          -0.69808296, -2.06112892, -1.44790449],
         [ 1.03944288,  0.13871728,  0.91478479,  0.47517105,  1.24638374,
           0.98666841,  0.49403488,  1.26101127],
         [-1.03949343, -0.39291108,  1.39107512,  1.73779253,  0.91656129,
           0.103381  ,  0.956243  ,  0.44743548]],

        [[ 0.79028054,  0.64244228, -0.37997334, -0.09130215, -2.3903429 ,
           0.71919208, -0.14079786,  0.98304272],
         [ 1.14678457,  1.58825227,  0.17137367, -0.62121819, -0.36103113,
          -0.04452576, -0.0886136 , -1.32884721],
         [ 0.06728957, -0.29701304, -0.52754207, -1.5785875 ,  1.47354834,
          -0.28545156,  0.49874194,  0.10277613],
         [-0.10117571, -1.34902427, -1.40789327,  0.09853599,  0.60420022,
           0.54869115, -0.49067696,  0.26696793],
         [ 1.11780279, -0.77929016,  1.13772094,  0.14374057,  0.33199688,
          -0.54057374, -0.45718861,  1.1577623 ],
         [-1.4005645 ,  1.15870496,  0.39292003,  0.88379515,  0.06440974,
           0.65013063,  0.03759244,  0.18730126]]],


       [[[-2.28272906,  0.06056305,  0.73632597,  0.10063274, -1.27497525,
          -0.95597581, -0.22745785,  0.40146498],
         [-1.37783475,  1.66000653, -1.80071745, -0.11805539, -0.27160583,
           0.30691418,  2.62243232, -1.95274516],
         [ 1.61364148,  1.91470546, -1.51984424,  2.13598224, -0.23156685,
          -0.74203698,  0.65316888,  0.08018846],
         [-1.8912854 , -0.50106158,  0.94937966, -0.10930541,  0.82136627,
          -1.33209063,  1.43371302, -1.36370916],
         [-0.52737928, -0.0681305 , -0.63472587,  0.41979229, -0.57093624,
          -0.15968764, -1.005951  , -2.06873572],
         [-2.34089346,  1.02593977,  0.90183415,  0.09504819,  0.53185448,
           1.11305345,  1.290016  , -1.76216646]],

        [[-0.10885459, -0.57089742, -0.55340708, -1.94445884,  1.30130049,
           0.6333372 , -1.03100083,  0.0111167 ],
         [ 0.59678149, -0.67601521, -1.25288718, -0.10922251,  3.06808996,
          -1.46701513, -0.42140765,  1.12485412],
         [ 1.21301567, -1.43304957, -0.56047239,  0.20716087,  1.40737646,
          -0.08386437, -0.21916043,  0.85692906],
         [ 1.59992399, -1.37044315, -0.71884386,  2.61830979, -0.74305496,
          -0.32021174,  1.43275058, -0.3891857 ],
         [-0.41355145,  0.22589689,  0.33154415,  0.86146815, -1.66326091,
           0.37581697, -3.2250516 , -0.48807863],
         [-2.52968957,  0.95801598, -1.20118154,  0.01141421, -0.11871498,
           0.04555184,  1.3950473 ,  0.37887998]]]]), array([[ 0.,  0.,  0.,  7.,  5.],
       [ 1.,  6.,  2.,  7.,  5.],
       [ 1.,  3.,  1.,  6.,  4.],
       [ 0.,  3.,  3.,  3.,  3.]])]
creating: createRoiPooling
output of m is : [[[[ 1.19822323  1.60344636  0.36516821]
   [ 1.39107513  1.73779249  1.26101124]]

  [[ 1.58825231  1.47354829  0.98304272]
   [ 1.158705    1.13772094  1.15776229]]]


 [[[ 1.43371308  1.43371308  0.08018846]
   [ 1.29001606  1.29001606 -1.7621665 ]]

  [[ 1.43275058  1.43275058  0.85692906]
   [ 1.39504731  1.39504731  0.37887999]]]


 [[[ 2.13598228  0.30691418  2.62243223]
   [ 0.82136625  0.82136625  1.43371308]]

  [[ 3.06808996  3.06808996 -0.08386437]
   [ 2.61830974  0.37581697  1.43275058]]]


 [[[-0.06152321 -0.06152321 -0.06152321]
   [-0.06152321 -0.06152321 -0.06152321]]

  [[ 0.09853599  0.09853599  0.09853599]
   [ 0.09853599  0.09853599  0.09853599]]]]

```

