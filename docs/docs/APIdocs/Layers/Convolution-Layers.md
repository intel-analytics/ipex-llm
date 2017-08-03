## SpatialConvolution ##

**Scala:**
```scala
val m = SpatialConvolution(nInputPlane,nOutputPlane,kernelW,kernelH,strideW=1,strideH=1,padW=0,padH=0,nGroup=1,propagateBack=true,wRegularizer=null,bRegularizer=null,initWeight=null, initBias=null, initGradWeight=null, initGradBias=null)
```
**Python:**
```python
m = SpatialConvolution(n_input_plane,n_output_plane,kernel_w,kernel_h,stride_w=1,stride_h=1,pad_w=0,pad_h=0,n_group=1,propagate_back=True,wRegularizer=None,bRegularizer=None,init_weight=None,init_bias=None,init_grad_weight=None,init_grad_bias=None)
```

SpatialConvolution is a module that applies a 2D convolution over an input image.

The input tensor in `forward(input)` is expected to be
either a 4D tensor (`batch x nInputPlane x height x width`) or a 3D tensor (` nInputPlane x height x width`). The convolution is performed on the last two dimensions.

Detailed paramter explaination for the constructor.
 
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
in forward(input) is expected to be a 4D tensor (nInputPlane x time x height x width).

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
 The input tensor in forward(input) is expected to be
 a 3D tensor (nInputPlane x height x width).

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
either a 4D tensor (`batch x nInputPlane x height x width`) or a 3D tensor (`nInputPlane x height x width`). The convolution is performed on the last two dimensions. `adjW` and `adjH` are used to adjust the size of the output image. The size of output tensor of `forward` will be :
```
  output width  = (width  - 1) * dW - 2*padW + kW + adjW
  output height = (height - 1) * dH - 2*padH + kH + adjH
``` 

Note, scala API also accepts a table input with two tensors: `T(convInput, sizeTensor)` where `convInput` is the standard input tensor, and the size of `sizeTensor` is used to set the size of the output (will ignore the `adjW` and `adjH` values used to construct the module). Use `SpatialFullConvolution[Table, T](...)` instead of `SpatialFullConvolution[Tensor,T](...)`) for table input.
 
This module can also be used without a bias by setting parameter `noBias = true` while constructing the module.
 
Other frameworks may call this operation "In-network Upsampling", "Fractionally-strided convolution", "Backwards Convolution," "Deconvolution", or "Upconvolution."
 
Reference: Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015: 3431-3440.

Detailed explaination of arguments in constructor. 

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
The input tensor in `forward(input)` is expected to be a 2D tensor
(`nInputFrame` x `inputFrameSize`) or a 3D tensor
(`nBatchFrame` x `nInputFrame` x `inputFrameSize`).

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

If input is a 4D tensor nInputPlane x depth x height x width,
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

