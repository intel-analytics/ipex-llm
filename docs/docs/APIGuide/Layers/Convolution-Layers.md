## SpatialConvolution ##

**Scala:**
```scala
val m = SpatialConvolution(nInputPlane,nOutputPlane,kernelW,kernelH,strideW=1,strideH=1,padW=0,padH=0,nGroup=1,propagateBack=true,wRegularizer=null,bRegularizer=null,initWeight=null, initBias=null, initGradWeight=null, initGradBias=null, withBias=true, dataFormat=DataFormat.NCHW)
```
**Python:**
```python
m = SpatialConvolution(n_input_plane,n_output_plane,kernel_w,kernel_h,stride_w=1,stride_h=1,pad_w=0,pad_h=0,n_group=1,propagate_back=True,wRegularizer=None,bRegularizer=None,init_weight=None,init_bias=None,init_grad_weight=None,init_grad_bias=None, with_bias=True, data_format="NCHW")
```

SpatialConvolution is a module that applies a 2D convolution over an input image.

The input tensor in `forward(input)` is expected to be
either a 4D tensor (`batch x nInputPlane x height x width`) or a 3D tensor (` nInputPlane x height x width`). The convolution is performed on the last two dimensions.
output of `forward(input)` is also expected to be a 4D tensor (`batch x outputPlane x height x width`)
or a 3D tensor (`outputPlane x height x width`)..

As for padding, when padW and padH are both -1, we use a padding algorithm similar to the "SAME" padding of tensorflow. That is
```scala
 outHeight = Math.ceil(inHeight.toFloat/strideH.toFloat)
 outWidth = Math.ceil(inWidth.toFloat/strideW.toFloat)

 padAlongHeight = Math.max(0, (outHeight - 1) * strideH + kernelH - inHeight)
 padAlongWidth = Math.max(0, (outWidth - 1) * strideW + kernelW - inWidth)

 padTop = padAlongHeight / 2
 padLeft = padAlongWidth / 2
```

Detailed parameter explanation for the constructor.
 
 * `nInputPlane` The number of expected input planes in the image given into forward()
 * `nOutputPlane` The number of output planes the convolution layer will produce.
 * `kernelW` The kernel width of the convolution
 * `kernelH` The kernel height of the convolution
 * `strideW` The step of the convolution in the width dimension.
 * `strideH` The step of the convolution in the height dimension
 * `padW`  padding to be added to width to the input.
 * `padH` padding to be added to height to the input.
 * `nGroup` Kernel group number
 * `propagateBack` whether to propagate gradient back
 * `wRegularizer` regularizer on weight. an instance of [[Regularizer]] (e.g. L1 or L2)
 * `bRegularizer` regularizer on bias. an instance of [[Regularizer]] (e.g. L1 or L2).
 * `initWeight` weight initializer
 * `initBias`  bias initializer
 * `initGradWeight` weight gradient initializer
 * `initGradBias` bias gradient initializer
 * `with_bias` the optional initial value for if need bias
 * `data_format` a string value (or DataFormat Object in Scala) of "NHWC" or "NCHW" to specify the input data format of this layer. In "NHWC" format
                        data is stored in the order of \[batch_size, height, width, channels\], in "NCHW" format data is stored
                        in the order of \[batch_size, channels, height, width\].
 
**Scala example:**
```scala

scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.Storage

val m = SpatialConvolution(2,1,2,2,1,1,0,0)
m.setInitMethod(weightInitMethod = BilinearFiller, biasInitMethod = Zeros)
val params = m.getParameters()

scala> print(params)
(1.0
0.0
0.0
0.0
1.0
0.0
0.0
0.0
0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 9],0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 9])

scala>
val input = Tensor(1,2,3,3).randn()
val output = m.forward(input)
val gradOut = Tensor(1,1,2,2).fill(0.2f)
val gradIn = m.backward(input,gradOut)

scala> print(input)
(1,1,.,.) =
-0.37011376     0.13565119      -0.73574775
-0.19486316     -0.4430604      -0.62543416
0.7017611       -0.6441595      -1.2953792

(1,2,.,.) =
-0.9903588      0.5669722       0.2630131
0.03392942      -0.6984676      -0.12389368
0.78704715      0.5411976       -1.3877676

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x2x3x3]

scala> print(output)
(1,1,.,.) =
-1.3604726      0.70262337
-0.16093373     -1.141528

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x2x2]

scala> print(gradOut)
(1,1,.,.) =
0.2     0.2
0.2     0.2

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x1x2x2]

scala> print(gradIn)
(1,1,.,.) =
0.2     0.2     0.0
0.2     0.2     0.0
0.0     0.0     0.0

(1,2,.,.) =
0.2     0.2     0.0
0.2     0.2     0.0
0.0     0.0     0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x3x3]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input = np.random.rand(1,3,3,3)
print "input is :",input

m = SpatialConvolution(3,1,2,2,1,1,0,0)
out = m.forward(input)
print "output m is :",out

grad_out = np.random.rand(1,1,2,2)
print "grad out of m is :",grad_out
grad_in = m.backward(input,grad_out)
print "grad input of m is :",grad_in
```
Gives the output,

```python
input is : [[[[ 0.75276617  0.44212513  0.90275949]
   [ 0.78205279  0.77864714  0.83647254]
   [ 0.76220944  0.22106036  0.68762202]]

  [[ 0.37346971  0.31532213  0.33276243]
   [ 0.69872884  0.07262236  0.66372462]
   [ 0.47803013  0.80194459  0.53313873]]

  [[ 0.56196833  0.20599878  0.47575818]
   [ 0.35454298  0.96910557  0.36234704]
   [ 0.64017738  0.95762579  0.50073035]]]]
creating: createSpatialConvolution
output m is : [[[[-1.08398974 -0.67615652]
   [-0.77027249 -0.82885492]]]]
grad out of m is : [[[[ 0.38295452  0.77048361]
   [ 0.11671955  0.76357513]]]]
grad input of m is : [[[[-0.02344826 -0.06515953 -0.03618064]
   [-0.06770924 -0.22586647 -0.14004168]
   [-0.01845866 -0.13653883 -0.10325129]]

  [[-0.09294108 -0.14361492  0.08727306]
   [-0.09885897 -0.21209857  0.29151234]
   [-0.02149716 -0.10957514  0.20318349]]

  [[-0.05926216 -0.04542646  0.14849319]
   [-0.09506465 -0.34244278 -0.03763583]
   [-0.02346931 -0.1815301  -0.18314059]]]]
```

---
## VolumetricConvolution ##

**Scala:**
```scala
val module = VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH,
  dT=1, dW=1, dH=1, padT=0, padW=0, padH=0, withBias=true, wRegularizer=null, bRegularizer=null)
```
**Python:**
```python
module = VolumetricConvolution(n_input_plane, n_output_plane, k_t, k_w, k_h,
  d_t=1, d_w=1, d_h=1, pad_t=0, pad_w=0, pad_h=0, with_bias=true, wRegularizer=null, bRegularizer=null)
```

Applies a 3D convolution over an input image composed of several input planes. The input tensor
in forward(input) is expected to be a 5D tensor (`batch x nInputPlane x depth(time) x height x width`) or
a 4D tensor (`nInputPlane x depth x height x width`).
Output of forward(input) is also expected to be a 5D tensor (`batch x depth(time) x outputPlane x height x width`) or
a 4D tensor (`outputPlane x depth x height x width`).
As for padding, when padW,padH, padT are all -1, we use a padding algorithm similar to the "SAME" padding of tensorflow.

* `nInputPlane` The number of expected input planes in the image given into forward()
* `nOutputPlane` The number of output planes the convolution layer will produce.
* `kT` The kernel size of the convolution in time
* `kW` The kernel width of the convolution
* `kH` The kernel height of the convolution
* `dT` The step of the convolution in the time dimension. Default is 1
* `dW` The step of the convolution in the width dimension. Default is 1
* `dH` The step of the convolution in the height dimension. Default is 1
* `padT` Additional zeros added to the input plane data on both sides of time axis.
         Default is 0. `(kT-1)/2` is often used here.
* `padW` The additional zeros added per width to the input planes.
* `padH` The additional zeros added per height to the input planes.
* `withBias` whether with bias.
* `wRegularizer` instance of [[Regularizer]]
                   (eg. L1 or L2 regularization), applied to the input weights matrices.
* `bRegularizer` instance of [[Regularizer]]
                   applied to the bias.
 
**Scala example:**
```scala
val layer = VolumetricConvolution(2, 3, 2, 2, 2, dT=1, dW=1, dH=1,
  padT=0, padW=0, padH=0, withBias=true)
val input = Tensor(2, 2, 2, 2).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.54846555      0.5549177
0.43748873      0.6596535

(1,2,.,.) =
0.87915933      0.5955469
0.67464 0.40921077

(2,1,.,.) =
0.24127467      0.49356017
0.6707502       0.5421975

(2,2,.,.) =
0.007834963     0.08188637
0.51387626      0.7376101

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x2x2]

layer.forward(input)
res16: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-0.6680023

(2,1,.,.) =
0.41926455

(3,1,.,.) =
-0.029196609

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x1x1x1]
```

**Python example:**
```python
layer = VolumetricConvolution(2, 3, 2, 2, 2, d_t=1, d_w=1, d_h=1,
          pad_t=0, pad_w=0, pad_h=0, with_bias=True, init_method="default",
          bigdl_type="float")
input = np.random.rand(2,2,2,2)
 array([[[[ 0.47639062,  0.76800312],
         [ 0.28834351,  0.21883535]],

        [[ 0.86097919,  0.89812597],
         [ 0.43632181,  0.58004824]]],


       [[[ 0.65784027,  0.34700039],
         [ 0.64511955,  0.1660241 ]],

        [[ 0.36060054,  0.71265665],
         [ 0.51755249,  0.6508298 ]]]])
 
layer.forward(input)
array([[[[ 0.54268712]]],


       [[[ 0.17670505]]],


       [[[ 0.40953237]]]], dtype=float32)

```

---
## SpatialDilatedConvolution ##

**Scala:**
```scala
val layer = SpatialDilatedConvolution(
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

The input tensor in `forward(input)` is expected to be
either a 4D tensor (`batch x nInputPlane x height x width`) or a 3D tensor (` nInputPlane x height x width`).
output of `forward(input)` is also expected to be a 4D tensor (`batch x outputPlane x height x width`)
or a 3D tensor (`outputPlane x height x width`).

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
 * `owidth  = floor(width + 2 * padW - dilationW * (kW-1) - 1) / dW + 1`
 * `oheight = floor(height + 2 * padH - dilationH * (kH-1) - 1) / dH + 1`

Reference Paper:
> Yu F, Koltun V. Multi-scale context aggregation by dilated convolutions[J].
arXiv preprint arXiv:1511.07122, 2015.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val layer = SpatialDilatedConvolution(1, 1, 2, 2, 1, 1, 0, 0, 2, 2)
val input = Tensor(T(T(
  T(1.0f, 2.0f, 3.0f, 4.0f),
  T(5.0f, 6.0f, 7.0f, 8.0f),
  T(9.0f, 1.0f, 2.0f, 3.0f),
  T(4.0f, 5.0f, 6.0f, 7.0f)
)))
val filter = Tensor(T(T(T(
  T(1.0f, 1.0f),
  T(1.0f, 1.0f)
))))
layer.weight.copy(filter)
layer.bias.zero()
layer.forward(input)
layer.backward(input, Tensor(T(T(
  T(0.1f, 0.2f),
  T(0.3f, 0.4f)
))))
```
Gives the output,

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
Gives the output,

```
array([[[ 15.,  10.],
        [ 22.,  26.]]], dtype=float32)
        
array([[[ 0.1       ,  0.2       ,  0.1       ,  0.2       ],
        [ 0.30000001,  0.40000001,  0.30000001,  0.40000001],
        [ 0.1       ,  0.2       ,  0.1       ,  0.2       ],
        [ 0.30000001,  0.40000001,  0.30000001,  0.40000001]]], dtype=float32)

```

---
## SpatialShareConvolution ##

**Scala:**
```scala
val layer = SpatialShareConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)
```
**Python:**
```python
layer = SpatialShareConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
```

 Applies a 2D convolution over an input image composed of several input planes.
 The input tensor in `forward(input)` is expected to be
 either a 4D tensor (`batch x nInputPlane x height x width`) or a 3D tensor (` nInputPlane x height x width`).
 output of `forward(input)` is also expected to be a 4D tensor (`batch x outputPlane x height x width`)
 or a 3D tensor (`outputPlane x height x width`).

 This layer has been optimized to save memory. If using this layer to construct multiple convolution
 layers, please add sharing script for the fInput and fGradInput. Please refer to the ResNet example.

**Scala example:**
```scala

    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    import com.intel.analytics.bigdl.nn._
    import com.intel.analytics.bigdl.tensor._

    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = SpatialShareConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1,
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1,
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor(Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor(Storage(biasData), 1, Array(nOutputPlane)))

    val input = Tensor(Storage(inputData), 1, Array(3, 1, 3, 4))
    val output = layer.updateOutput(input)
 
    > output
res2: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
49.0    63.0    38.0
91.0    105.0   56.0

(2,1,.,.) =
49.0    63.0    38.0
91.0    105.0   56.0

(3,1,.,.) =
49.0    63.0    38.0
91.0    105.0   56.0
```

**Python example:**
```python
nInputPlane = 1
nOutputPlane = 1
kW = 2
kH = 2
dW = 1
dH = 1
padW = 0
padH = 0
layer = SpatialShareConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

input = np.array([
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1,
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1,
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1]
    ).astype("float32").reshape(3, 1, 3, 4)
layer.forward(input)

> print (output)
array([[[[-3.55372381, -4.0352459 , -2.65861344],
         [-4.99829054, -5.4798131 , -3.29477644]]],


       [[[-3.55372381, -4.0352459 , -2.65861344],
         [-4.99829054, -5.4798131 , -3.29477644]]],


       [[[-3.55372381, -4.0352459 , -2.65861344],
         [-4.99829054, -5.4798131 , -3.29477644]]]], dtype=float32)
```

---
## SpatialFullConvolution ##

**Scala:**
```scala
val m  = SpatialFullConvolution(nInputPlane, nOutputPlane, kW, kH, dW=1, dH=1, padW=0, padH=0, adjW=0, adjH=0,nGroup=1, noBias=false,wRegularizer=null,bRegularizer=null)
```
**Python:**
```python
m = SpatialFullConvolution(n_input_plane,n_output_plane,kw,kh,dw=1,dh=1,pad_w=0,pad_h=0,adj_w=0,adj_h=0,n_group=1,no_bias=False,init_method='default',wRegularizer=None,bRegularizer=None)
```

SpatialFullConvolution is a module that applies a 2D full convolution over an input image. 

The input tensor in `forward(input)` is expected to be
either a 4D tensor (`batch x nInputPlane x height x width`) or a 3D tensor (` nInputPlane x height x width`).
output of `forward(input)` is also expected to be a 4D tensor (`batch x outputPlane x height x width`)
or a 3D tensor (`outputPlane x height x width`).
The convolution is performed on the last two dimensions. `adjW` and `adjH` are used to adjust the size of the output image. The size of output tensor of `forward` will be :
```
  output width  = (width  - 1) * dW - 2*padW + kW + adjW
  output height = (height - 1) * dH - 2*padH + kH + adjH
``` 

Note, scala API also accepts a table input with two tensors: `T(convInput, sizeTensor)` where `convInput` is the standard input tensor, and the size of `sizeTensor` is used to set the size of the output (will ignore the `adjW` and `adjH` values used to construct the module). Use `SpatialFullConvolution[Table, T](...)` instead of `SpatialFullConvolution[Tensor,T](...)`) for table input.
 
This module can also be used without a bias by setting parameter `noBias = true` while constructing the module.
 
Other frameworks may call this operation "In-network Upsampling", "Fractionally-strided convolution", "Backwards Convolution," "Deconvolution", or "Upconvolution."
 
Reference: Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015: 3431-3440.

Detailed explanation of arguments in constructor. 

 * `nInputPlane` The number of expected input planes in the image given into forward()
 * `nOutputPlane` The number of output planes the convolution layer will produce.
 * `kW` The kernel width of the convolution.
 * `kH` The kernel height of the convolution.
 * `dW` The step of the convolution in the width dimension. Default is 1.
 * `dH` The step of the convolution in the height dimension. Default is 1.
 * `padW` The additional zeros added per width to the input planes. Default is 0.
 * `padH` The additional zeros added per height to the input planes. Default is 0.
 * `adjW` Extra width to add to the output image. Default is 0.
 * `adjH` Extra height to add to the output image. Default is 0.
 * `nGroup` Kernel group number.
 * `noBias` If bias is needed.
 * `wRegularizer` instance of [[Regularizer]]
                   (eg. L1 or L2 regularization), applied to the input weights matrices.
 * `bRegularizer` instance of [[Regularizer]]
                   applied to the bias.
 
**Scala example:**

Tensor Input example: 

```scala

scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.Storage

val m = SpatialFullConvolution(1, 2, 2, 2, 1, 1,0, 0, 0, 0, 1, false)

val input = Tensor(1,1,3,3).randn()
val output = m.forward(input)
val gradOut = Tensor(1,2,4,4).fill(0.1f)
val gradIn = m.backward(input,gradOut)

scala> print(input)
(1,1,.,.) =
0.18219171      1.3252861       -1.3991559
0.82611334      1.0313315       0.6075537
-0.7336061      0.3156875       -0.70616096

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x1x3x3]

scala> print(output)
(1,1,.,.) =
-0.49278542     -0.5823938      -0.8304068      -0.077556044
-0.5028842      -0.7281958      -1.1927067      -0.34262076
-0.41680115     -0.41400516     -0.7599415      -0.42024887
-0.5286566      -0.30015367     -0.5997892      -0.32439864

(1,2,.,.) =
-0.13131973     -0.5770084      1.1069719       -0.6003375
-0.40302444     -0.07293816     -0.2654545      0.39749345
0.37311426      -0.49090374     0.3088816       -0.41700447
-0.12861171     0.09394867      -0.17229918     0.05556257

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x4x4]

scala> print(gradOut)
(1,1,.,.) =
0.1     0.1     0.1     0.1
0.1     0.1     0.1     0.1
0.1     0.1     0.1     0.1
0.1     0.1     0.1     0.1

(1,2,.,.) =
0.1     0.1     0.1     0.1
0.1     0.1     0.1     0.1
0.1     0.1     0.1     0.1
0.1     0.1     0.1     0.1

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x2x4x4]

scala> print(gradIn)
(1,1,.,.) =
-0.05955213     -0.05955213     -0.05955213
-0.05955213     -0.05955213     -0.05955213
-0.05955213     -0.05955213     -0.05955213

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x3x3]

	
```

Table input Example
```scala

scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.{T, Table}

val m = SpatialFullConvolution(1, 2, 2, 2, 1, 1,0, 0, 0, 0, 1, false)

val input1 = Tensor(1, 3, 3).randn()
val input2 = Tensor(3, 3).fill(2.0f)
val input = T(input1, input2)
val output = m.forward(input)
val gradOut = Tensor(2,4,4).fill(0.1f)
val gradIn = m.backward(input,gradOut)

scala> print(input)
 {
        2: 2.0  2.0     2.0
           2.0  2.0     2.0
           2.0  2.0     2.0
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]
        1: (1,.,.) =
           1.276177     0.62761325      0.2715257
           -0.030832397 0.5046206       0.6835176
           -0.5832693   0.17266633      0.7461992

           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x3x3]
 }

scala> print(output)
(1,.,.) =
-0.18339296     0.04208675      -0.17708774     -0.30901802
-0.1484881      0.23592418      0.115615785     -0.11288056
-0.47266048     -0.41772115     0.07501307      0.041751802
-0.4851033      -0.5427048      -0.18293871     -0.12682784

(2,.,.) =
0.6391188       0.845774        0.41208875      0.13754106
-0.45785713     0.31221163      0.6006259       0.36563575
-0.24076991     -0.31931365     0.31651747      0.4836449
0.24247466      -0.16731171     -0.20887817     0.19513035

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x4]

scala> print(gradOut)
(1,.,.) =
0.1     0.1     0.1     0.1
0.1     0.1     0.1     0.1
0.1     0.1     0.1     0.1
0.1     0.1     0.1     0.1

(2,.,.) =
0.1     0.1     0.1     0.1
0.1     0.1     0.1     0.1
0.1     0.1     0.1     0.1
0.1     0.1     0.1     0.1

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x4x4]

scala> print(gradIn)
 {
        2: 0.0  0.0     0.0
           0.0  0.0     0.0
           0.0  0.0     0.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
        1: (1,.,.) =
           0.16678208   0.16678208      0.16678208
           0.16678208   0.16678208      0.16678208
           0.16678208   0.16678208      0.16678208

           [com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x3]
 }
 
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

m = SpatialFullConvolution(1, 2, 2, 2, 1, 1,0, 0, 0, 0, 1, False)

print "--------- tensor input---------"
tensor_input = np.random.rand(1,3,3)
print "input is :",tensor_input
out = m.forward(tensor_input)
print "output m is :",out

print "----------- table input --------"
adj_input=np.empty([3,3])
adj_input.fill(2.0)
table_input = [tensor_input,adj_input]
print "input is :",table_input
out = m.forward(table_input)
print "output m is :",out
```
Gives the output,

```python
creating: createSpatialFullConvolution
--------- tensor input---------
input is : [[[  9.03998497e-01   4.43054896e-01   6.19571211e-01]
  [  4.24573060e-01   3.29886286e-04   5.48427154e-02]
  [  8.99004782e-01   3.25514441e-01   6.85294650e-01]]]
output m is : [[[-0.04712385  0.21949144  0.0843184   0.14336972]
  [-0.28748769  0.39192575  0.00372696  0.27235305]
  [-0.16292028  0.41943201  0.03476509  0.18813471]
  [-0.28051955  0.29929382 -0.0689255   0.28749463]]

 [[-0.21336153 -0.35994443 -0.29239666 -0.38612381]
  [-0.33000433 -0.41727966 -0.36827195 -0.34524575]
  [-0.2410759  -0.38439807 -0.27613443 -0.39401439]
  [-0.38188276 -0.36746511 -0.37627563 -0.34141305]]]
----------- table input --------
input is : [array([[[  9.03998497e-01,   4.43054896e-01,   6.19571211e-01],
        [  4.24573060e-01,   3.29886286e-04,   5.48427154e-02],
        [  8.99004782e-01,   3.25514441e-01,   6.85294650e-01]]]), array([[ 2.,  2.,  2.],
       [ 2.,  2.,  2.],
       [ 2.,  2.,  2.]])]
output m is : [[[-0.04712385  0.21949144  0.0843184   0.14336972]
  [-0.28748769  0.39192575  0.00372696  0.27235305]
  [-0.16292028  0.41943201  0.03476509  0.18813471]
  [-0.28051955  0.29929382 -0.0689255   0.28749463]]

 [[-0.21336153 -0.35994443 -0.29239666 -0.38612381]
  [-0.33000433 -0.41727966 -0.36827195 -0.34524575]
  [-0.2410759  -0.38439807 -0.27613443 -0.39401439]
  [-0.38188276 -0.36746511 -0.37627563 -0.34141305]]]
```

---
## SpatialSeparableConvolution ##

**Scala:**
```scala
val m  = SpatialSeparableConvolution(nInputChannel, nOutputChannel, depthMultiplier, kW, kH, sW = 1, sH = 1, pW = 0, pH = 0, hasBias = True, dataFormat = DataFormat.NCHW, wRegularizer = null, bRegularizer = null, pRegularizer = null, initDepthWeight = null, initPointWeight = null, initBias = null)
```
**Python:**
```python
m = SpatialSeparableConvolution(n_input_channel, n_output_channel, depth_multiplier, kernel_w, kernel_h, stride_w=1, stride_h=1, pad_w=0, pad_h=0, with_bias=True, data_format="NCHW", w_regularizer=None, b_regularizer=None, p_regularizer=None)
```

Separable convolutions consist in first performing a depthwise spatial convolution (which acts
on each input channel separately) followed by a pointwise convolution which mixes together the
resulting output channels. The  depthMultiplier argument controls how many output channels are
enerated per input channel in the depthwise step.

 * `nInputChannel` The number of expected input planes in the image given into forward()
 * `nOutputChannel` The number of output planes the convolution layer will produce.
 * `depthMultiplier` how many internal channels are generated per input channel
 * `kW` The kernel width of the convolution.
 * `kH` The kernel height of the convolution.
 * `sW` The step of the convolution in the width dimension.
 * `sH` The step of the convolution in the height dimension.
 * `pW` The additional zeros added per width to the input planes. Default is 0.
 * `pH` The additional zeros added per height to the input planes. Default is 0.
 * `hasBias` do we use a bias on the output, default is true
 * `dataFormat` image data format, which can be NHWC or NCHW, default value is NCHW
 * `wRegularizer` kernel parameter regularizer
 * `bRegularizer` bias regularizer
 * `pRegularizer` point wise kernel parameter regularizer
 * `initDepthWeight` kernel parameter init tensor
 * `initPointWeight` point wise kernel parameter init tensor
 * `initBias` bias init tensor
 
**Scala example:**

```scala

scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat

val m = SpatialSeparableConvolution[Float](1, 2, 1, 2, 2, dataFormat = DataFormat.NCHW)
val input = Tensor(1, 1, 3, 3).randn()
val output = m.forward(input)
val gradOut = Tensor(1, 2, 2, 2).fill(0.1f)
val gradIn = m.backward(input,gradOut)

scala> print(input)
(1,1,.,.) =
-0.6636712      -1.3765892      -1.51044
0.4502934       -0.38438025     -0.4279503
-1.5327895      -0.33594692     1.5972415

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x1x3x3]

scala> print(output)
(1,1,.,.) =
-0.2903078      -0.5241474
-0.17961408     -0.11239494

(1,2,.,.) =
-1.3147768      -2.3738143
-0.81345534     -0.5090261

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x2x2]

scala> print(gradOut)
(1,1,.,.) =
0.1     0.1
0.1     0.1

(1,2,.,.) =
0.1     0.1
0.1     0.1

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x2x2x2]

scala> print(gradIn)
(1,1,.,.) =
0.088415675     0.17780215      0.08938648
0.15242647      0.26159728      0.109170794
0.06401079      0.08379511      0.019784318

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x3x3]
```


**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

m = SpatialSeparableConvolution(1, 2, 1, 2, 2)
tensor_input = np.random.rand(1, 1, 3, 3)
print "input is :",tensor_input
out = m.forward(tensor_input)
print "output m is :",out
```
Gives the output,

```python
creating: createSpatialFullConvolution
input is : [[[[ 0.77269038  0.82476003  0.58228669]
   [ 0.35123569  0.25496535  0.16736527]
   [ 0.62138293  0.83156875  0.77565037]]]]
output m is : [[[[ 0.91489887  0.81591743]
   [ 0.84698057  0.76615578]]

  [[ 1.05583775  0.94160837]
   [ 0.97745675  0.88418102]]]]
```

---
## SpatialConvolutionMap ##

**Scala:**
```scala
val layer = SpatialConvolutionMap(
  connTable,
  kW,
  kH,
  dW = 1,
  dH = 1,
  padW = 0,
  padH = 0,
  wRegularizer = null,
  bRegularizer = null)
```
**Python:**
```python
layer = SpatialConvolutionMap(
  conn_table,
  kw,
  kh,
  dw=1,
  dh=1,
  pad_w=0,
  pad_h=0,
  wRegularizer=None,
  bRegularizer=None)
```

This class is a generalization of SpatialConvolution.
It uses a generic connection table between input and output features.
The SpatialConvolution is equivalent to using a full connection table.  
A Connection Table is the mapping of input/output feature map, stored in a 2D Tensor. The first column is the input feature maps. The second column is output feature maps.


Full Connection table:
```scala
val conn = SpatialConvolutionMap.full(nin: Int, nout: In)
```

One to One connection table:
```scala
val conn = SpatialConvolutionMap.oneToOne(nfeat: Int)
```

Random Connection table:
```scala
val conn = SpatialConvolutionMap.random(nin: Int, nout: Int, nto: Int)
```


**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val conn = SpatialConvolutionMap.oneToOne(3)
```
`conn` is
```
conn: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0	1.0
2.0	2.0
3.0	3.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]
```

```
val module = SpatialConvolutionMap(SpatialConvolutionMap.oneToOne(3), 2, 2)

pritnln(module.forward(Tensor.range(1, 48, 1).resize(3, 4, 4)))
```
Gives the output,
```
com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
4.5230045	5.8323975	7.1417904
9.760576	11.069969	12.379362
14.998148	16.30754	17.616934

(2,.,.) =
-5.6122046	-5.9227824	-6.233361
-6.8545156	-7.165093	-7.4756703
-8.096827	-8.407404	-8.71798

(3,.,.) =
13.534529	13.908197	14.281864
15.029203	15.402873	15.77654
16.523876	16.897545	17.271214

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3x3]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = SpatialConvolutionMap(np.array([(1, 1), (2, 2), (3, 3)]), 2, 2)

print(module.forward(np.arange(1, 49, 1).reshape(3, 4, 4)))
```
Gives the output,
```
[array([[[-1.24280548, -1.70889318, -2.17498088],
        [-3.10715604, -3.57324386, -4.03933144],
        [-4.97150755, -5.43759441, -5.90368223]],

       [[-5.22062826, -5.54696751, -5.87330723],
        [-6.52598572, -6.85232496, -7.17866373],
        [-7.8313427 , -8.15768337, -8.48402214]],

       [[ 0.5065825 ,  0.55170798,  0.59683061],
        [ 0.68707776,  0.73219943,  0.77732348],
        [ 0.86757064,  0.91269422,  0.95781779]]], dtype=float32)]
```

---
## TemporalConvolution ##

**Scala:**
```scala
val module = TemporalConvolution(
  inputFrameSize, outputFrameSize, kernelW, strideW = 1, propagateBack = true,
  wRegularizer = null, bRegularizer = null, initWeight = null, initBias = null,
  initGradWeight = null, initGradBias = null
  )
```
**Python:**
```python
module = TemporalConvolution(
  input_frame_size, output_frame_size, kernel_w, stride_w = 1, propagate_back = True,
  w_regularizer = None, b_regularizer = None, init_weight = None, init_bias = None,
  init_grad_weight = None, init_grad_bias = None
  )
```

 Applies a 1D convolution over an input sequence composed of nInputFrame frames.
 The input tensor in `forward(input)` is expected to be a 3D tensor
 (`nBatchFrame` x `nInputFrame` x `inputFrameSize`) or a 2D tensor
 (`nInputFrame` x `inputFrameSize`).
 Output of `forward(input)` is expected to be a 3D tensor
 (`nBatchFrame` x `nOutputFrame` x `outputFrameSize`) or a 2D tensor
 (`nOutputFrame` x `outputFrameSize`).

 * `inputFrameSize` The input frame size expected in sequences given into `forward()`.
 * `outputFrameSize` The output frame size the convolution layer will produce.
 * `kernelW` The kernel width of the convolution
 * `strideW` The step of the convolution in the width dimension.
 * `propagateBack` Whether propagate gradient back, default is true.
 * `wRegularizer` instance of `Regularizer`
                     (eg. L1 or L2 regularization), applied to the input weights matrices.
 * `bRegularizer` instance of `Regularizer`
                     applied to the bias.
 * `initWeight` Initial weight
 * `initBias` Initial bias
 * `initGradWeight` Initial gradient weight
 * `initGradBias` Initial gradient bias
 * `T` The numeric type in the criterion, usually which are `Float` or `Double`
 
**Scala example:**
```scala
import com.intel.analytics.bigdl.numeric.NumericFloat
val seed = 100
RNG.setSeed(seed)
val inputFrameSize = 5
val outputFrameSize = 3
val kW = 5
val dW = 2
val layer = TemporalConvolution(inputFrameSize, outputFrameSize, kW, dW)

Random.setSeed(seed)
val input = Tensor(10, 5).apply1(e => Random.nextFloat())
val gradOutput = Tensor(3, 3).apply1(e => Random.nextFloat())

val output = layer.updateOutput(input)
> println(output)
2017-07-21 06:18:00 INFO  ThreadPool$:79 - Set mkl threads to 1 on thread 1
-0.34987333	-0.0063185245	-0.45821175	
-0.20838472	0.15102878	-0.5656665	
-0.13935827	-0.099345684	-0.76407385	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

val gradInput = layer.updateGradInput(input, gradOutput)
> println(gradInput)
0.018415622	-0.10201519	-0.15641063	-0.08271551	-0.060939234	
0.13609992	0.14934899	0.06083451	-0.13943195	-0.11092151	
-0.14552939	-0.024670592	-0.29887137	-0.14555064	-0.05840567	
0.09920926	0.2705848	0.016875947	-0.27233958	-0.069991685	
-0.0024300043	-0.15160085	-0.20593905	-0.2894306	-0.057458147	
0.06390554	0.07710219	0.105445914	-0.26714328	-0.18871497	
0.13901645	-0.10651534	0.006758575	-0.08754986	-0.13747974	
-0.026543075	-0.044046614	0.13146847	-0.01198944	-0.030542556	
0.18396454	-0.055985756	-0.03506116	-0.02156017	-0.09211717	
0.0	0.0	0.0	0.0	0.0	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 10x5]
```

**Python example:**
```python
from bigdl.nn.layer import TemporalConvolution
import numpy as np
inputFrameSize = 5
outputFrameSize = 3
kW = 5
dW = 2
layer = TemporalConvolution(inputFrameSize, outputFrameSize, kW, dW)

input = np.random.rand(10, 5)
gradOutput = np.random.rand(3, 3)

output = layer.forward(input)
> print(output)
[[ 0.43262666  0.52964264 -0.09026626]
 [ 0.46828389  0.3391096   0.04789509]
 [ 0.37985104  0.13899082 -0.05767119]]
 
gradInput = layer.backward(input, gradOutput)
> print(gradInput)
[[-0.08801709  0.03619258  0.06944641 -0.01570761  0.00682773]
 [-0.02754797  0.07474414 -0.08249797  0.04756897  0.0096445 ]
 [-0.14383194  0.05168077  0.27049363  0.10419817  0.05263135]
 [ 0.12452157 -0.02296585  0.14436334  0.02482709 -0.12260982]
 [ 0.04890725 -0.19043611  0.2909058  -0.10708418  0.07759682]
 [ 0.05745121  0.10499261  0.02989995  0.13047372  0.09119483]
 [-0.09693538 -0.12962547  0.22133902 -0.09149387  0.29208034]
 [ 0.2622599  -0.12875232  0.21714815  0.11484481 -0.00040091]
 [ 0.07558989  0.00072951  0.12860702 -0.27085134  0.10740379]
 [ 0.          0.          0.          0.          0.        ]]

```

---
## TemporalMaxPooling

**scala:**
```scala
val m = TemporalMaxPooling(k_w, d_w = k_w)
```

```python
m = TemporalMaxPooling(k_w, d_w = k_w)
```

Applies 1D max-pooling operation in `k_w` regions by step size `d_w` steps.
Input sequence composed of nInputFrame frames.
The input tensor in forward(input) is expected to be a 2D tensor
(nInputFrame x inputFrameSize) or a 3D tensor (nBatchFrame x nInputFrame x inputFrameSize).

If the input sequence is a 2D tensor of dimension nInputFrame x inputFrameSize,
the output sequence will be nOutputFrame x inputFrameSize where

```
nOutputFrame = (nInputFrame - k_w) / d_w + 1
```

  * k_w: kernel width
  * d_w: step size in width

```scala
scala>
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

val module = TemporalMaxPooling(4)
val input = Tensor(1, 8, 5).rand()
val output = module.forward(input)
val gradOutput = Tensor(1, 2, 5).rand()
val gradInput = module.backward(input, gradOutput)

scala>
println(output)
(1,.,.) =
0.6248109817970544	0.7783127573784441	0.8484677821397781	0.6721713887527585	0.9674506767187268	
0.9587726043537259	0.8359494411852211	0.6541860734578222	0.7671433456707746	0.8246882800012827	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x5]

scala>
println(gradInput)
(1,.,.) =
0.0	0.0	0.0	0.0	0.012729122070595622	
0.0	0.1717955127824098	0.00636984477750957	0.0	0.0	
0.0	0.0	0.0	0.24560829368419945	0.0	
0.8350501179229468	0.0	0.0	0.0	0.0	
0.0	0.9017464134376496	0.662078354973346	0.4239895506761968	0.0	
0.09446275723166764	0.0	0.0	0.0	0.974747731583193	
0.0	0.0	0.0	0.0	0.0	
0.0	0.0	0.0	0.0	0.0	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x8x5]
```

```python
from bigdl.nn.layer import *
import numpy as np

module = TemporalMaxPooling(4)
input = np.random.rand(1, 8, 5)
output = module.forward(input)
grad_output = np.random.rand(1, 2, 5)
grad_input = module.backward(input, gradOutput)

print "output is :",output
print "gradient input m is :",grad_input
```

```
creating: createTemporalMaxPooling
output is : [[[0.6248109817970544	0.7783127573784441	0.8484677821397781	0.6721713887527585	0.9674506767187268]	
[0.9587726043537259	0.8359494411852211	0.6541860734578222	0.7671433456707746	0.8246882800012827]]]	
gradient input m is : [[[0.0	0.0	0.0	0.0	0.012729122070595622]	
[0.0	0.1717955127824098	0.00636984477750957	0.0	0.0]	
[0.0	0.0	0.0	0.24560829368419945	0.0	]
[0.8350501179229468	0.0	0.0	0.0	0.0	]
[0.0	0.9017464134376496	0.662078354973346	0.4239895506761968	0.0]	
[0.09446275723166764	0.0	0.0	0.0	0.974747731583193]	
[0.0	0.0	0.0	0.0	0.0]	
[0.0	0.0	0.0	0.0	0.0]]]	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x8x5]
```

---
## VolumetricFullConvolution ##

**Scala:**
```scala
val m  = VolumetricFullConvolution(
  nInputPlane, nOutputPlane,
  kT, kW, kH,
  dT, dW = 1, dH = 1,
  padT = 0, padW = 0, padH = 0,
  adjT = 0, adjW = 0, adjH = 0,
  nGroup=1, noBias=false,wRegularizer=null,bRegularizer=null)
```
**Python:**
```python
m = VolumetricFullConvolution(
    n_input_plane, n_output_plane,
    kt, kw, kh, 
    dt=1, dw=1,dh=1,
    pad_t=0, pad_w=0, pad_h=0, 
    adj_t=0, adj_w=0,adj_h=0,
    n_group=1,no_bias=False,init_method='default',wRegularizer=None,bRegularizer=None)
```

`VolumetricFullConvolution` Apply a 3D full convolution over an 3D input image, a sequence of images, or a video etc.
The input tensor is expected to be a 4D or 5D(with batch) tensor. Note that instead
of setting adjT, adjW and adjH, `VolumetricConvolution` also accepts a table input
with two tensors: T(convInput, sizeTensor) where convInput is the standard input tensor,
and the size of sizeTensor is used to set the size of the output (will ignore the adjT, adjW and
adjH values used to construct the module). This module can be used without a bias by setting
parameter noBias = true while constructing the module.

Applies a 3D convolution over an input image composed of several input planes. The input tensor
in forward(input) is expected to be a 5D tensor (`batch x nInputPlane x depth(time) x height x width`) or
a 4D tensor (`nInputPlane x depth x height x width`).
Output of forward(input) is also expected to be a 5D tensor (`batch x depth(time) x outputPlane x height x width`) or
a 4D tensor (`outputPlane x depth x height x width`).

```
odepth  = (depth  - 1) * dT - 2*padT + kT + adjT
owidth  = (width  - 1) * dW - 2*padW + kW + adjW
oheight = (height - 1) * dH - 2*padH + kH + adjH
```

Other frameworks call this operation "In-network Upsampling", "Fractionally-strided convolution",
"Backwards Convolution," "Deconvolution", or "Upconvolution."

Reference Paper: Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic
segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
2015: 3431-3440.

 * nInputPlane The number of expected input planes in the image given into forward()
 * nOutputPlane The number of output planes the convolution layer will produce.
 * kT The kernel depth of the convolution.
 * kW The kernel width of the convolution.
 * kH The kernel height of the convolution.
 * dT The step of the convolution in the depth dimension. Default is 1.
 * dW The step of the convolution in the width dimension. Default is 1.
 * dH The step of the convolution in the height dimension. Default is 1.
 * padT The additional zeros added per depth to the input planes. Default is 0.
 * padW The additional zeros added per width to the input planes. Default is 0.
 * padH The additional zeros added per height to the input planes. Default is 0.
 * adjT Extra depth to add to the output image. Default is 0.
 * adjW Extra width to add to the output image. Default is 0.
 * adjH Extra height to add to the output image. Default is 0.
 * nGroup Kernel group number.
 * noBias If bias is needed.
 * wRegularizer: instance of `Regularizer`
 *             (eg. L1 or L2 regularization), applied to the input weights matrices.
 * bRegularizer: instance of `Regularizer`
                   applied to the bias.
 
**Scala example:**

Tensor Input example: 

```scala
scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val m = VolumetricFullConvolution(2, 1, 2, 2, 2)

val input = Tensor(1, 2, 2, 3, 3).randn()
val output = m.forward(input)
val gradOut = Tensor(1, 1, 3, 4, 4).fill(0.2f)
val gradIn = m.backward(input, gradOut)

scala> println(input)
(1,1,1,.,.) =
0.3903321	-0.90453357	1.735308	
-1.2824814	-0.27802613	-0.3977802	
-0.08534186	0.6385388	-0.86845094	

(1,1,2,.,.) =
-0.24652982	0.69465446	0.1713606	
0.07106233	-0.88137305	1.0625362	
-0.553569	1.1822331	-2.2488093	

(1,2,1,.,.) =
0.552869	0.4108489	1.7802315	
0.018191056	0.72422534	-0.6423254	
-0.4077748	0.024120487	-0.42820823	

(1,2,2,.,.) =
-1.3711191	-0.37988988	-2.1587164	
-0.85155743	-1.5785019	-0.77727056	
0.42253423	0.79593533	0.15303874	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x2x2x3x3]

scala> println(output)
(1,1,1,.,.) =
-0.29154167	-0.027156994	-0.6949123	-0.22638178	
0.091479614	-0.106284864	-0.23198327	-0.5334093	
0.092822656	-0.13807209	-0.07207352	-0.023272723	
-0.19217497	-0.18892932	-0.089907974	-0.059967346	

(1,1,2,.,.) =
0.08078699	-0.0242998	0.27271587	0.48551774	
-0.30726838	0.5497404	-0.7220843	0.48132813	
0.007951438	-0.39301366	0.56711966	-0.39552623	
-0.016941413	-0.5530351	0.21254264	-0.22647215	

(1,1,3,.,.) =
-0.38189644	-0.5241636	-0.49781954	-0.59505236	
-0.23887709	-0.99911994	-0.773817	-0.63575095	
-0.1193203	0.016682416	-0.41216886	-0.5211964	
-0.06341652	-0.32541442	0.43984014	-0.16862796	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x3x4x4]

scala> println(gradOut)
(1,1,1,.,.) =
0.2	0.2	0.2	0.2	
0.2	0.2	0.2	0.2	
0.2	0.2	0.2	0.2	
0.2	0.2	0.2	0.2	

(1,1,2,.,.) =
0.2	0.2	0.2	0.2	
0.2	0.2	0.2	0.2	
0.2	0.2	0.2	0.2	
0.2	0.2	0.2	0.2	

(1,1,3,.,.) =
0.2	0.2	0.2	0.2	
0.2	0.2	0.2	0.2	
0.2	0.2	0.2	0.2	
0.2	0.2	0.2	0.2	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x1x3x4x4]
scala> println(gradIn)
(1,1,1,.,.) =
-0.089189366	-0.089189366	-0.089189366	
-0.089189366	-0.089189366	-0.089189366	
-0.089189366	-0.089189366	-0.089189366	

(1,1,2,.,.) =
-0.089189366	-0.089189366	-0.089189366	
-0.089189366	-0.089189366	-0.089189366	
-0.089189366	-0.089189366	-0.089189366	

(1,2,1,.,.) =
0.06755526	0.06755526	0.06755526	
0.06755526	0.06755526	0.06755526	
0.06755526	0.06755526	0.06755526	

(1,2,2,.,.) =
0.06755526	0.06755526	0.06755526	
0.06755526	0.06755526	0.06755526	
0.06755526	0.06755526	0.06755526	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x2x3x3]

```

Table input Example
```scala
scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T

val m = VolumetricFullConvolution(1, 2, 2, 2, 2)

val input1 = Tensor(1, 3, 3, 3).randn()
val input2 = Tensor(3, 3, 3).fill(2.0f)
val input = T(input1, input2)
val output = m.forward(input)
val gradOut = Tensor(2, 4, 4, 4).fill(0.1f)
val gradIn = m.backward(input, gradOut)

scala> println(input)
{
  2: (1,.,.) =
  2.0	2.0	2.0
  2.0	2.0	2.0
  2.0	2.0	2.0

  (2,.,.) =
  2.0	2.0	2.0
  2.0	2.0	2.0
  2.0	2.0	2.0

  (3,.,.) =
  2.0	2.0	2.0
  2.0	2.0	2.0
  2.0	2.0	2.0

  [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3x3]
  1: (1,1,.,.) =
  0.23809154	1.2167819	0.3664989
  0.8797001	1.5262067	0.15420714
  0.38004395	-0.24190372	-1.1151218

  (1,2,.,.) =
  -1.895742	1.8554556	0.62502027
  -0.6004498	0.056441266	-0.66499823
  0.7039313	-0.08569297	-0.08191566

  (1,3,.,.) =
  -1.9555066	-0.20133287	-0.22135374
  0.8918014	-1.2684877	0.14211883
  2.5802526	1.1118578	-1.3165624

  [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x3x3x3]
}

scala> println(output)
(1,1,.,.) =
-0.2578445	-0.48271507	-0.28246504	-0.20139077
-0.43916196	-0.72301924	-0.2915339	-0.20471849
-0.41347015	-0.36456454	0.021684423	-0.20852578
-0.255981	-0.17165771	-0.04553239	-0.19543594

(1,2,.,.) =
0.18660262	-0.8204256	-0.08807768	-0.1023551
0.026309028	-0.49442527	0.3699256	-0.12729678
-0.34651133	0.08542377	0.24221262	-0.47949657
-0.29622912	-0.15598825	-0.23278731	-0.32802662

(1,3,.,.) =
0.6303606	-1.0451282	0.21740273	-0.03673452
-0.039471984	-0.2264648	0.15774214	-0.30815765
-1.0726243	-0.13914594	0.08537227	-0.30611742
-0.55404246	-0.29725668	-0.037192106	-0.20331946

(1,4,.,.) =
0.19113302	-0.68506914	-0.21211714	-0.26207167
-0.40826926	0.068062216	-0.5962198	-0.18985644
-0.7111124	0.3466564	0.2185097	-0.5388211
-0.16902745	0.10249108	-0.09487718	-0.35127735

(2,1,.,.) =
-0.2744591	-0.21165672	-0.17422867	-0.25680506
-0.24608877	-0.1242196	-0.02206999	-0.23146236
-0.27057967	-0.17076656	-0.18083718	-0.35417527
-0.28634468	-0.24118122	-0.30961025	-0.41247135

(2,2,.,.) =
-0.41682464	-0.5772195	-0.159199	-0.2294753
-0.41187716	-0.41886678	0.4104582	-0.1382559
-0.08818802	0.459113	0.48080307	-0.3373265
-0.18515268	-0.14088067	-0.67644227	-0.67253566

(2,3,.,.) =
-0.009801388	-0.83997947	-0.39409852	-0.29002026
-0.6333371	-0.66267097	0.52607954	-0.10082486
-0.46748784	-0.08717018	-0.54928875	-0.59819674
-0.103552	0.22147804	-0.20562811	-0.46321797

(2,4,.,.) =
0.090245515	-0.28537494	-0.24673338	-0.289634
-0.98199505	-0.7408645	-0.4654177	-0.35744694
-0.5410351	-0.48618284	-0.40212065	-0.26319134
0.4081596	0.8880725	-0.26220837	-0.73146355

  [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x4x4]

scala> println(gradOut)
(1,1,.,.) =
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1

(1,2,.,.) =
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1

(1,3,.,.) =
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1

(1,4,.,.) =
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1

(2,1,.,.) =
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1

(2,2,.,.) =
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1

(2,3,.,.) =
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1

(2,4,.,.) =
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1
0.1	0.1	0.1	0.1

  [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x4x4x4]

scala> println(gradIn)
{
  2: (1,.,.) =
  0.0	0.0	0.0
  0.0	0.0	0.0
  0.0	0.0	0.0

  (2,.,.) =
  0.0	0.0	0.0
  0.0	0.0	0.0
  0.0	0.0	0.0

  (3,.,.) =
  0.0	0.0	0.0
  0.0	0.0	0.0
  0.0	0.0	0.0

  [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3x3]
  1: (1,1,.,.) =
  0.048819613	0.048819613	0.048819613
  0.048819613	0.048819613	0.048819613
  0.048819613	0.048819613	0.048819613

  (1,2,.,.) =
  0.048819613	0.048819613	0.048819613
  0.048819613	0.048819613	0.048819613
  0.048819613	0.048819613	0.048819613

  (1,3,.,.) =
  0.048819613	0.048819613	0.048819613
  0.048819613	0.048819613	0.048819613
  0.048819613	0.048819613	0.048819613

  [com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x3x3]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

m = VolumetricFullConvolution(2, 1, 2, 2, 2)

print "--------- tensor input---------"
tensor_input = np.random.rand(1, 2, 2, 3, 3)
print "input is :",tensor_input
out = m.forward(tensor_input)
print "output m is :",out

print "----------- table input --------"
adj_input=np.empty([3, 3, 3])
adj_input.fill(2.0)
table_input = [tensor_input,adj_input]
print "input is :",table_input
out = m.forward(table_input)
print "output m is :",out
```

```
creating: createVolumetricFullConvolution
--------- tensor input---------
input is : [[[[[ 0.41632522  0.62726142  0.11133406]
    [ 0.61013369  0.76320391  0.27937597]
    [ 0.3596402   0.85087329  0.18706284]]

   [[ 0.19224562  0.79333622  0.02064112]
    [ 0.34019388  0.36193739  0.0189533 ]
    [ 0.01245767  0.59638721  0.97882726]]]


  [[[ 0.03641869  0.92804035  0.08934243]
    [ 0.96598196  0.54331079  0.9157464 ]
    [ 0.31659511  0.48128023  0.13775686]]

   [[ 0.44624135  0.02830871  0.95668413]
    [ 0.32971474  0.46466264  0.58239329]
    [ 0.94129846  0.27284845  0.59931096]]]]]
output m is : [[[[[ 0.24059629  0.11875484 -0.07601731  0.18490529]
    [ 0.17978033 -0.05925606 -0.06877603 -0.00254188]
    [ 0.33574528  0.10908454 -0.01606898  0.22380096]
    [ 0.24050319  0.17277193  0.10569186  0.20417407]]

   [[ 0.26733595  0.26336247 -0.16927747  0.04417276]
    [ 0.39058518 -0.08025722 -0.11981271  0.08441451]
    [ 0.21994853 -0.1127445  -0.01282334 -0.25795668]
    [ 0.34960991  0.17045188  0.0885388   0.08292522]]

   [[ 0.29700345  0.22094724  0.27189076  0.07538646]
    [ 0.27829763  0.01766421  0.32052374 -0.09809484]
    [ 0.28885722  0.08438809  0.24915564 -0.08578731]
    [ 0.25339472 -0.09679155  0.09070791  0.21198538]]]]]
----------- table input --------
input is : [array([[[[[ 0.41632522,  0.62726142,  0.11133406],
          [ 0.61013369,  0.76320391,  0.27937597],
          [ 0.3596402 ,  0.85087329,  0.18706284]],

         [[ 0.19224562,  0.79333622,  0.02064112],
          [ 0.34019388,  0.36193739,  0.0189533 ],
          [ 0.01245767,  0.59638721,  0.97882726]]],


        [[[ 0.03641869,  0.92804035,  0.08934243],
          [ 0.96598196,  0.54331079,  0.9157464 ],
          [ 0.31659511,  0.48128023,  0.13775686]],

         [[ 0.44624135,  0.02830871,  0.95668413],
          [ 0.32971474,  0.46466264,  0.58239329],
          [ 0.94129846,  0.27284845,  0.59931096]]]]]), array([[[ 2.,  2.,  2.],
        [ 2.,  2.,  2.],
        [ 2.,  2.,  2.]],

       [[ 2.,  2.,  2.],
        [ 2.,  2.,  2.],
        [ 2.,  2.,  2.]],

       [[ 2.,  2.,  2.],
        [ 2.,  2.,  2.],
        [ 2.,  2.,  2.]]])]
output m is : [[[[[ 0.24059629  0.11875484 -0.07601731  0.18490529]
    [ 0.17978033 -0.05925606 -0.06877603 -0.00254188]
    [ 0.33574528  0.10908454 -0.01606898  0.22380096]
    [ 0.24050319  0.17277193  0.10569186  0.20417407]]

   [[ 0.26733595  0.26336247 -0.16927747  0.04417276]
    [ 0.39058518 -0.08025722 -0.11981271  0.08441451]
    [ 0.21994853 -0.1127445  -0.01282334 -0.25795668]
    [ 0.34960991  0.17045188  0.0885388   0.08292522]]

   [[ 0.29700345  0.22094724  0.27189076  0.07538646]
    [ 0.27829763  0.01766421  0.32052374 -0.09809484]
    [ 0.28885722  0.08438809  0.24915564 -0.08578731]
    [ 0.25339472 -0.09679155  0.09070791  0.21198538]]]]]
```

---
## LocallyConnected1D ##

**Scala:**
```scala
val module = LocallyConnected1D(
  nInputFrame,inputFrameSize, outputFrameSize, kernelW, strideW = 1, propagateBack = true,
  wRegularizer = null, bRegularizer = null, initWeight = null, initBias = null,
  initGradWeight = null, initGradBias = null)
```
**Python:**
```python
module = LocallyConnected1D(
  n_input_frame, input_frame_size, output_frame_size, kernel_w, stride_w=1, propagate_back=True,
  w_regularizer=None, b_regularizer=None, init_weight=None, init_bias=None,
  init_grad_weight=None, init_grad_bias=None)
```

 Applies a 1D convolution over an input sequence composed of nInputFrame frames with unshared weights.
 The input tensor in `forward(input)` is expected to be a 3D tensor
 (`nBatchFrame` x `nInputFrame` x `inputFrameSize`) or a 2D tensor
 (`nInputFrame` x `inputFrameSize`).
 Output of `forward(input)` is expected to be a 3D tensor
 (`nBatchFrame` x `nOutputFrame` x `outputFrameSize`) or a 2D tensor
 (`nOutputFrame` x `outputFrameSize`).

* `nInputFrame` Length of the input frame expected in sequences given into `forward()`.
* `inputFrameSize` The input frame size expected in sequences given into `forward()`.
* `outputFrameSize` The output frame size the convolution layer will produce.
* `kernelW` The kernel width of the convolution
* `strideW` The step of the convolution in the width dimension.
* `propagateBack` Whether propagate gradient back, default is true.
* `wRegularizer` instance of `Regularizer`
                 (eg. L1 or L2 regularization), applied to the input weights matrices.
* `bRegularizer` instance of `Regularizer`
                 applied to the bias.
* `initWeight` Initial weight
* `initBias` Initial bias
* `initGradWeight` Initial gradient weight
* `initGradBias` Initial gradient bias
* `T` The numeric type in the criterion, usually which are `Float` or `Double`
 
**Scala example:**
```scala
import com.intel.analytics.bigdl.numeric.NumericFloat
val seed = 100
RNG.setSeed(seed)
val nInputFrame = 10
val inputFrameSize = 5
val outputFrameSize = 3
val kW = 5
val dW = 2
val layer = LocallyConnected1D(nInputFrame, inputFrameSize, outputFrameSize, kW, dW)

Random.setSeed(seed)
val input = Tensor(10, 5).apply1(e => Random.nextFloat())
val gradOutput = Tensor(3, 3).apply1(e => Random.nextFloat())

val output = layer.updateOutput(input)
> println(output)
(1,.,.) =
-0.2896616	0.018883035	-0.45641226	
-0.41183263	-0.33292565	0.27988705	
0.076636955	-0.39710814	0.59631383	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

val gradInput = layer.updateGradInput(input, gradOutput)
> println(gradInput)
(1,.,.) =
0.018415622	 -0.10201519  -0.15641063	-0.08271551	  -0.060939234	
0.13609992	 0.14934899	  0.06083451	-0.13943195	  -0.11092151	
-0.08760113	 0.06923811	  -0.07376863	0.06743649	  0.042455398	
0.064692274	 0.15720972	  0.13673763	0.03617531	  0.12507091	
-0.078272685 -0.25193688  0.10712688	-0.11330205	  -0.19239372	
-0.10032463	 -0.06266674  0.1048636	    0.26058376	  -0.40386787	
-0.10379471	 0.07291742	  -0.28790376   0.06023993	  0.057165086	
0.15167418	 0.07384029	  -0.052450493  -0.07709345	  -0.016432922	
-0.1044948	 0.060714033  0.08341185	-0.082587965  0.052750245	
0.0	         0.0	      0.0	        0.0	          0.0	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 10x5]
```

**Python example:**
```python
from bigdl.nn.layer import LocallyConnected1D
import numpy as np
nInputFrame = 10
inputFrameSize = 5
outputFrameSize = 3
kW = 5
dW = 2
layer = LocallyConnected1D(nInputFrame, inputFrameSize, outputFrameSize, kW, dW)

input = np.random.rand(10, 5)
gradOutput = np.random.rand(3, 3)

output = layer.forward(input)
> print(output)
[[ 0.37944531 -0.25905907 -0.02284177]
 [-0.06727666 -0.48430425 -0.12338555]
 [ 0.5237388  -0.72521925 -0.21979821]]
 
gradInput = layer.backward(input, gradOutput)
> print(gradInput)
[[-0.22256926 -0.11267932  0.05445758 -0.06569604  0.00799843]
 [ 0.08402308  0.00340014  0.04202492 -0.05055574  0.11835655]
 [ 0.00352848 -0.02568576 -0.08056175  0.06994451  0.09152003]
 [ 0.04089724 -0.19517297  0.19212601 -0.21531224  0.03563112]
 [-0.28906721  0.07873128 -0.01326483 -0.18504807  0.02452871]
 [-0.09979478 -0.1009931  -0.25594842  0.14314197 -0.30875987]
 [-0.00814501 -0.02431242 -0.1140819  -0.14522757 -0.09230929]
 [-0.11231296  0.0053857   0.00582423  0.18309449  0.13369997]
 [-0.01302226 -0.13035376  0.02006471  0.09794775 -0.08067283]
 [ 0.          0.          0.          0.          0.        ]]

```

---
## LocallyConnected2D ##
**Scala:**
```scala
val module = LocallyConnected2D(
   nInputPlane, inputWidth, inputHeight, nOutputPlane, kernelW, kernelH, 
   strideW = 1, strideH = 1, padW = 0, padH = 0, propagateBack = true, 
   wRegularizer = null, bRegularizer = null, initWeight = null, initBias = null,
   initGradWeight = null, initGradBias = null, withBias = true, format = DataFormat.NCHW)
```
**Python:**
```python
module = LocallyConnected2D(
    n_input_plane, input_width, input_height, n_output_plane,
    kernel_w, kernel_h, stride_w=1, stride_h=1, pad_w=0, pad_h=0,
    propagate_back=True, wRegularizer=None, bRegularizer=None,
    init_weight=None, init_bias=None, init_grad_weight=None,
    init_grad_bias=None, with_bias=True, data_format="NCHW")
```                            

The LocallyConnected2D layer works similarly to the [SpatialConvolution](#spatialconvolution) layer, except that weights are unshared, that is, a different set of filters is applied at each different patch of the input.

* `nInputPlane` The number of expected input planes in the image.
* `inputWidth` The input width.
* `inputHeight` The input height.
* `nOutputPlane` The number of output planes the convolution layer will produce.
* `kernelW` The kernel width of the convolution.
* `kernelH` The kernel height of the convolution.
* `strideW` The step of the convolution in the width dimension.
* `strideH` The step of the convolution in the height dimension.
* `padW` The additional zeros added per width to the input planes.
* `padH` The additional zeros added per height to the input planes.
* `propagateBack` Whether to propagate gradient back.
* `wRegularizer` Weight regularizer.
* `bRegularizer` Bias regularizer.
* `initWeight` Initial weight.
* `initBias` Initial bias.
* `initGradWeight` Initial gradient weight.
* `initGradBias` Initial gradient bias.
* `withBias` Whether to include bias.
* `format` Data format of the input. Either "NHWC" or "NCHW".

                
**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.LocallyConnected2D
import com.intel.analytics.bigdl.tensor.Tensor

val layer = LocallyConnected2D(2, 6, 3, 3, 1, 2, format=DataFormat.NHWC)
val input = Tensor(1, 3, 6, 2).rand()
val output = layer.forward(input)

input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.580334	0.59774524
0.35452667	0.9134508
0.56355035	0.10698065
0.95197415	0.10339011
0.6571263	0.35572186
0.31106102	0.97996104

(1,2,.,.) =
0.87887615	0.8108329
0.7184107	0.487163
0.85714895	0.30265027
0.4407469	0.94804007
0.5460197	0.01421738
0.74672765	0.23766468

(1,3,.,.) =
0.10655104	0.008004449
0.142883	0.7885532
0.12025218	0.9536053
0.85908693	0.088657066
0.42529714	0.64380044
0.8999299	0.6074533

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x3x6x2]


output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.7493179	  0.27513236	-0.2982489
-0.41126582	  0.21310717	0.36723173
-0.039210618  0.13379198	-0.28216434
0.19143593	  -0.61731964	-0.018212453
0.24316064	  -1.1187351	0.74201244
0.060099036	  -0.5223875	-0.95892024

(1,2,.,.) =
-0.4977209	 0.19270697	   -0.00647337
-0.18642347	 -0.057786018  0.33848432
0.044415057	 -0.12975587   -0.054034393
0.46163	     0.06908426	   -0.17127737
-0.07933617	 0.190754	   0.6044696
-0.723027	 0.14250416	   0.51286244

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x6x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.layer import LocallyConnected2D

layer = LocallyConnected2D(2, 6, 3, 3, 1, 2, data_format="NHWC")
input = np.random.rand(1, 3, 6, 2)
output = layer.forward(input)

print(input)
[[[[  6.13867469e-01,   5.15609721e-01],
   [  5.14951616e-01,   4.93308310e-01],
   [  7.34218405e-01,   6.06311945e-01],
   [  9.38263668e-01,   3.26766196e-01],
   [  4.24955447e-02,   3.30625440e-01],
   [  3.55858423e-01,   6.10869469e-01]],

  [[  3.75525334e-02,   4.93555936e-02],
   [  4.44188497e-01,   3.51001813e-02],
   [  8.11139320e-01,   4.87916727e-01],
   [  4.00786464e-01,   1.65522882e-01],
   [  5.98298525e-01,   9.54343135e-01],
   [  2.25942857e-01,   5.76090257e-02]],

  [[  1.34708024e-01,   4.81133433e-01],
   [  7.63198918e-01,   2.96906096e-01],
   [  6.01935030e-01,   2.39748841e-01],
   [  5.32036004e-01,   1.86107334e-01],
   [  9.38617798e-01,   6.83511632e-04],
   [  2.34639435e-01,   8.04904706e-01]]]]
  
print(output)
[[[[-0.01100884,  0.59226239, -0.15626255],
   [ 0.29099607,  0.16722232, -0.39429453],
   [ 0.22557285,  0.30368266,  0.53235221],
   [ 0.05602939, -0.07677993, -0.32399753],
   [ 0.47589377, -0.15926963,  0.1135996 ],
   [ 0.25957716,  0.17047183,  0.21640816]],

  [[-0.15497619,  0.29392233, -0.12167639],
   [ 0.60150111, -0.001901  ,  0.294438  ],
   [-0.05004713,  0.22379839,  0.53971994],
   [ 0.23204027,  0.17921877,  0.29594338],
   [ 0.91105354,  0.881271  , -0.69958985],
   [ 0.45518994, -0.645486  ,  0.37325871]]]]
```

---
## UpSampling1D ##

**Scala:**
```scala
val module = UpSampling1D(length: Int)
```
**Python:**
```python
m = UpSampling1D(length)
```
UpSampling layer for 1D inputs. Repeats each temporal step length times along the time axis. 

If input's size is (batch, steps, features), then the output's size will be (batch, steps * length, features). 

**Scala example:** 

```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T

val module = UpSampling1D(2)
val input = Tensor(2, 3, 3).range(1, 18)
module.forward(input)
```
The output should be 
```scala
res: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.0	2.0	3.0
1.0	2.0	3.0
4.0	5.0	6.0
4.0	5.0	6.0
7.0	8.0	9.0
7.0	8.0	9.0

(2,.,.) =
10.0	11.0	12.0
10.0	11.0	12.0
13.0	14.0	15.0
13.0	14.0	15.0
16.0	17.0	18.0
16.0	17.0	18.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x6x3]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input = np.random.rand(1,2,2,2,2)
print "input is :",input

m = UpSampling3D([2,2,2])
out = m.forward(input)
print "output m is :",out
```
Gives the output,

```python
input is : 
[[[[[ 0.80314148  0.79158609]
    [ 0.3988551   0.91726511]]
   [[ 0.86814757  0.90733343]
    [ 0.34470172  0.03056507]]]

  [[[ 0.62367481  0.20093996]
    [ 0.57614891  0.75442351]]
   [[ 0.52572424  0.04730832]
    [ 0.74973562  0.2245238 ]]]]]
creating: createUpSampling3D
output m is : 
[[[[[ 0.80314147  0.80314147  0.7915861   0.7915861 ]
    [ 0.80314147  0.80314147  0.7915861   0.7915861 ]
    [ 0.39885509  0.39885509  0.91726512  0.91726512]
    [ 0.39885509  0.39885509  0.91726512  0.91726512]]

   [[ 0.80314147  0.80314147  0.7915861   0.7915861 ]
    [ 0.80314147  0.80314147  0.7915861   0.7915861 ]
    [ 0.39885509  0.39885509  0.91726512  0.91726512]
    [ 0.39885509  0.39885509  0.91726512  0.91726512]]

   [[ 0.86814755  0.86814755  0.90733343  0.90733343]
    [ 0.86814755  0.86814755  0.90733343  0.90733343]
    [ 0.34470171  0.34470171  0.03056507  0.03056507]
    [ 0.34470171  0.34470171  0.03056507  0.03056507]]

   [[ 0.86814755  0.86814755  0.90733343  0.90733343]
    [ 0.86814755  0.86814755  0.90733343  0.90733343]
    [ 0.34470171  0.34470171  0.03056507  0.03056507]
    [ 0.34470171  0.34470171  0.03056507  0.03056507]]]


  [[[ 0.62367481  0.62367481  0.20093997  0.20093997]
    [ 0.62367481  0.62367481  0.20093997  0.20093997]
    [ 0.57614893  0.57614893  0.7544235   0.7544235 ]
    [ 0.57614893  0.57614893  0.7544235   0.7544235 ]]

   [[ 0.62367481  0.62367481  0.20093997  0.20093997]
    [ 0.62367481  0.62367481  0.20093997  0.20093997]
    [ 0.57614893  0.57614893  0.7544235   0.7544235 ]
    [ 0.57614893  0.57614893  0.7544235   0.7544235 ]]

   [[ 0.52572423  0.52572423  0.04730832  0.04730832]
    [ 0.52572423  0.52572423  0.04730832  0.04730832]
    [ 0.74973559  0.74973559  0.2245238   0.2245238 ]
    [ 0.74973559  0.74973559  0.2245238   0.2245238 ]]

   [[ 0.52572423  0.52572423  0.04730832  0.04730832]
    [ 0.52572423  0.52572423  0.04730832  0.04730832]
    [ 0.74973559  0.74973559  0.2245238   0.2245238 ]
    [ 0.74973559  0.74973559  0.2245238   0.2245238 ]]]]]
```

---
## UpSampling2D ##

**Scala:**
```scala
val module = UpSampling2D(size: Array[Int], format: DataFormat = DataFormat.NCHW)
```
**Python:**
```python
m = UpSampling2D(size, data_format)
```
UpSampling layer for 2D inputs. 
Repeats the heights and widths of the data by size[0] and size[1] respectively. 

If input's dataformat is NCHW, then the size of output will be (N, C, H * size[0], W * size[1]). 

Detailed parameter explanation for the constructor. 
 * `size` tuple of 2 integers. The upsampling factors for heights and widths.
 * `data_format` a string value (or DataFormat Object in Scala) of "NHWC" or "NCHW" to specify the input data format of this layer. In "NHWC" format
                        data is stored in the order of \[batch_size, height, width, channels\], in "NCHW" format data is stored
                        in the order of \[batch_size, channels, height, width\].

**Scala example:** 

```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T

val module = UpSampling2D(Array(2, 3))
val input = Tensor(2, 2, 3, 3).range(1, 36)
module.forward(input)
```
The output should be 
```scala
res: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
1.0	1.0	1.0	2.0	2.0	2.0	3.0	3.0	3.0
1.0	1.0	1.0	2.0	2.0	2.0	3.0	3.0	3.0
4.0	4.0	4.0	5.0	5.0	5.0	6.0	6.0	6.0
4.0	4.0	4.0	5.0	5.0	5.0	6.0	6.0	6.0
7.0	7.0	7.0	8.0	8.0	8.0	9.0	9.0	9.0
7.0	7.0	7.0	8.0	8.0	8.0	9.0	9.0	9.0

(1,2,.,.) =
10.0	10.0	10.0	11.0	11.0	11.0	12.0	12.0	12.0
10.0	10.0	10.0	11.0	11.0	11.0	12.0	12.0	12.0
13.0	13.0	13.0	14.0	14.0	14.0	15.0	15.0	15.0
13.0	13.0	13.0	14.0	14.0	14.0	15.0	15.0	15.0
16.0	16.0	16.0	17.0	17.0	17.0	18.0	18.0	18.0
16.0	16.0	16.0	17.0	17.0	17.0	18.0	18.0	18.0

(2,1,.,.) =
19.0	19.0	19.0	20.0	20.0	20.0	21.0	21.0	21.0
19.0	19.0	19.0	20.0	20.0	20.0	21.0	21.0	21.0
22.0	22.0	22.0	23.0	23.0	23.0	24.0	24.0	24.0
22.0	22.0	22.0	23.0	23.0	23.0	24.0	24.0	24.0
25.0	25.0	25.0	26.0	26.0	26.0	27.0	27....
```


**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = UpSampling2D([2, 3])
input = np.arange(1, 37).reshape(2, 2, 3, 3)
module.forward(input)
```
The output is 
```python
array([[[[  1.,   1.,   1.,   2.,   2.,   2.,   3.,   3.,   3.],
         [  1.,   1.,   1.,   2.,   2.,   2.,   3.,   3.,   3.],
         [  4.,   4.,   4.,   5.,   5.,   5.,   6.,   6.,   6.],
         [  4.,   4.,   4.,   5.,   5.,   5.,   6.,   6.,   6.],
         [  7.,   7.,   7.,   8.,   8.,   8.,   9.,   9.,   9.],
         [  7.,   7.,   7.,   8.,   8.,   8.,   9.,   9.,   9.]],

        [[ 10.,  10.,  10.,  11.,  11.,  11.,  12.,  12.,  12.],
         [ 10.,  10.,  10.,  11.,  11.,  11.,  12.,  12.,  12.],
         [ 13.,  13.,  13.,  14.,  14.,  14.,  15.,  15.,  15.],
         [ 13.,  13.,  13.,  14.,  14.,  14.,  15.,  15.,  15.],
         [ 16.,  16.,  16.,  17.,  17.,  17.,  18.,  18.,  18.],
         [ 16.,  16.,  16.,  17.,  17.,  17.,  18.,  18.,  18.]]],


       [[[ 19.,  19.,  19.,  20.,  20.,  20.,  21.,  21.,  21.],
         [ 19.,  19.,  19.,  20.,  20.,  20.,  21.,  21.,  21.],
         [ 22.,  22.,  22.,  23.,  23.,  23.,  24.,  24.,  24.],
         [ 22.,  22.,  22.,  23.,  23.,  23.,  24.,  24.,  24.],
         [ 25.,  25.,  25.,  26.,  26.,  26.,  27.,  27.,  27.],
         [ 25.,  25.,  25.,  26.,  26.,  26.,  27.,  27.,  27.]],

        [[ 28.,  28.,  28.,  29.,  29.,  29.,  30.,  30.,  30.],
         [ 28.,  28.,  28.,  29.,  29.,  29.,  30.,  30.,  30.],
         [ 31.,  31.,  31.,  32.,  32.,  32.,  33.,  33.,  33.],
         [ 31.,  31.,  31.,  32.,  32.,  32.,  33.,  33.,  33.],
         [ 34.,  34.,  34.,  35.,  35.,  35.,  36.,  36.,  36.],
         [ 34.,  34.,  34.,  35.,  35.,  35.,  36.,  36.,  36.]]]], dtype=float32)
```

---
## UpSampling3D ##

**Scala:**
```scala
val module = UpSampling3D(size: Array[Int])
```
**Python:**
```python
m = UpSampling3D(size)
```

UpSampling3D is a module that upsamples for 3D inputs.
It repeats the 1st, 2nd and 3rd dimensions of the data by size[0], size[1] and size[2] respectively.
The input data is assumed to be of the form ```minibatch x channels x depth x height x width```.

**Scala example:**
```scala

scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.Storage

val module = UpSampling3D(Array(2, 2, 2))
val input = Tensor(1, 2, 2, 2, 2).randn()
val output = module.forward(input)

> input
(1,1,1,.,.) =
0.8626614	-0.25849837
0.89711547	0.41256216

(1,1,2,.,.) =
0.031144595	0.28527617
0.36917794	-0.9892453

(1,2,1,.,.) =
-1.7768023	-0.39210165
1.9640301	-2.2383325

(1,2,2,.,.) =
0.41984457	-1.1820035
0.23327439	-0.17730176

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x2x2x2]

> output
(1,1,1,.,.) =
0.8626614	0.8626614	-0.25849837	-0.25849837
0.8626614	0.8626614	-0.25849837	-0.25849837
0.89711547	0.89711547	0.41256216	0.41256216
0.89711547	0.89711547	0.41256216	0.41256216

(1,1,2,.,.) =
0.8626614	0.8626614	-0.25849837	-0.25849837
0.8626614	0.8626614	-0.25849837	-0.25849837
0.89711547	0.89711547	0.41256216	0.41256216
0.89711547	0.89711547	0.41256216	0.41256216

(1,1,3,.,.) =
0.031144595	0.031144595	0.28527617	0.28527617
0.031144595	0.031144595	0.28527617	0.28527617
0.36917794	0.36917794	-0.9892453	-0.9892453
0.36917794	0.36917794	-0.9892453	-0.9892453

(1,1,4,.,.) =
0.031144595	0.031144595	0.28527617	0.28527617
0.031144595	0.031144595	0.28527617	0.28527617
0.36917794	0.36917794	-0.9892453	-0.9892453
0.36917794	0.36917794	-0.9892453	-0.9892453

(1,2,1,.,.) =
-1.7768023	-1.7768023	-0.39210165	-0.39210165
-1.7768023	-1.7768023	-0.39210165	-0.39210165
1.9640301	1.9640301	-2.2383325	-2.2383325
1.9640301	1.9640301	-2.2383325	-2.2383325

(1,2,2,.,.) =
-1.7768023	-1.7768023	-0.39210165	-0.39210165
-1.7768023	-1.7768023	-0.39210165	-0.39210165
1.9640301	1.9640301	-2.2383325	-2.2383325
1.9640301	1.9640301	-2.2383325	-2.2383325

(1,2,3,.,.) =
0.41984457	0.41984457	-1.1820035	-1.1820035
0.41984457	0.41984457	-1.1820035	-1.1820035
0.23327439	0.23327439	-0.17730176	-0.17730176
0.23327439	0.23327439	-0.17730176	-0.17730176

(1,2,4,.,.) =
0.41984457	0.41984457	-1.1820035	-1.1820035
0.41984457	0.41984457	-1.1820035	-1.1820035
0.23327439	0.23327439	-0.17730176	-0.17730176
0.23327439	0.23327439	-0.17730176	-0.17730176

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x4x4x4]

```

---
## ResizeBilinear ##

**Scala:**
```scala
val module = ResizeBilinear(outputHeight, outputWidth,
                      alignCorners=false, dataFormat = DataFormat.NCHW)
```
**Python:**
```python
m = ResizeBilinear(outputHeight, outputWidth,
               alignCorners=False, dataFormat="NCHW")
```

Resize the input image with bilinear interpolation. The input image must be a float tensor with
NHWC or NCHW layout.
 
**Scala example:**
```scala

scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val module = ResizeBilinear(4, 4)
val input = Tensor(1, 1, 2, 2).range(1, 4)
val output = module.forward(input)

> output
(1,1,.,.) =
1.0	1.5	2.0	2.0	
2.0	2.5	3.0	3.0	
3.0	3.5	4.0	4.0	
3.0	3.5	4.0	4.0	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x1x4x4]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = ResizeBilinear(4, 4)
input = np.arange(1, 5).reshape(1, 1, 2, 2)
output = module.forward(input)
print output
```
The output is 
```python
[[[[ 1.   1.5  2.   2. ]
   [ 2.   2.5  3.   3. ]
   [ 3.   3.5  4.   4. ]
   [ 3.   3.5  4.   4. ]]]]
```
