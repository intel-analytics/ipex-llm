## SpatialConvolution ##

**Scala:**
```scala
val m = SpatialConvolution(nInputPlane,nOutputPlane,kernelW,kernelH,strideW=1,strideH=1,padW=0,padH=0,nGroup=1,propagateBack=true,wRegularizer=null,bRegularizer=null,initWeight=null, initBias=null, initGradWeight=null, initGradBias=null)
```
**Python:**
```python
m = SpatialConvolution(n_input_plane,n_output_plane,kernel_w,kernel_h,stride_w=1,stride_h=1,pad_w=0,pad_h=0,n_group=1,propagate_back=True,init_method="default")
```

SpatialConvolution is a module that applies a 2D convolution over an input image.

The input tensor in `forward(input)` is expected to be
either a 4D tensor (`batch x nInputPlane x height x width`) or a 3D tensor (`batch x height x width`). The convolution is performed on the last two dimensions.

Detailed paramter explaination for the constructor.
 
 * param nInputPlane: The number of expected input planes in the image given into forward()
 * param nOutputPlane: The number of output planes the convolution layer will produce.
 * param kernelW: The kernel width of the convolution
 * param kernelH: The kernel height of the convolution
 * param strideW: The step of the convolution in the width dimension.
 * param strideH: The step of the convolution in the height dimension
 * param padW:  padding to be added to width to the input.
 * param padH: padding to be added to height to the input.
 * param nGroup: Kernel group number
 * param propagateBack: whether to propagate gradient back
 * param wRegularizer: regularizer on weight. an instance of [[Regularizer]] (e.g. L1 or L2)
 * param bRegularizer: regularizer on bias. an instance of [[Regularizer]] (e.g. L1 or L2).
 * param initWeight: weight initializer
 * param initBias:  bias initializer
 * param initGradWeight: weight gradient initializer
 * param initGradBias: bias gradient initializer
 
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
produces output:
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