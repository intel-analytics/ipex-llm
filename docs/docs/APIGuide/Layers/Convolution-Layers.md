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
or
```scala
val m = SpatialFullConvolution(InputPlane, nOutputPlane, kW, kH, dW=1, dH=1, padW=0, padH=0, adjW=0, adjH=0,nGroup=1, noBias=false,wRegularizer=null,bRegularizer=null)
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
  padH = 0)
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
  pad_h=0)
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

