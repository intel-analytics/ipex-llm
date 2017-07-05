## SpatialFullConvolution ##

**Scala:**
```scala
val m  = SpatialFullConvolution[Tensor[T], T](nInputPlane, nOutputPlane, kW, kH, dW=1, dH=1, padW=0, padH=0, adjW=0, adjH=0,nGroup=1, noBias=false,wRegularizer=null,bRegularizer=null)
```
or
```scala
val m = SpatialFullConvolution[Table, T](InputPlane, nOutputPlane, kW, kH, dW=1, dH=1, padW=0, padH=0, adjW=0, adjH=0,nGroup=1, noBias=false,wRegularizer=null,bRegularizer=null)
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

 * param nInputPlane The number of expected input planes in the image given into forward()
 * param nOutputPlane The number of output planes the convolution layer will produce.
 * param kW The kernel width of the convolution.
 * param kH The kernel height of the convolution.
 * param dW The step of the convolution in the width dimension. Default is 1.
 * param dH The step of the convolution in the height dimension. Default is 1.
 * param padW The additional zeros added per width to the input planes. Default is 0.
 * param padH The additional zeros added per height to the input planes. Default is 0.
 * param adjW Extra width to add to the output image. Default is 0.
 * param adjH Extra height to add to the output image. Default is 0.
 * param nGroup Kernel group number.
 * param noBias If bias is needed.
 * param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * param bRegularizer: instance of [[Regularizer]]
 *                    applied to the bias.
 
**Scala example:**

Tensor Input example: 

```scala

scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.Storage

val m = SpatialFullConvolution[Tensor[Float],Float](1, 2, 2, 2, 1, 1,0, 0, 0, 0, 1, false)

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

val m = SpatialFullConvolution[Table, Float](1, 2, 2, 2, 1, 1,0, 0, 0, 0, 1, false)

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
produces output:
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
