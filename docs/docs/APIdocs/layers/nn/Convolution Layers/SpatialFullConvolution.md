## SpatialFullConvolution ##

**Scala:**
```scala
val m  = SpatialFullConvolution[Tensor[T], T](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, adjW, adjH,nGroup, noBias,wRegularizer,bRegularizer)
```
or
```scala
val m = SpatialFullConvolution[Table, T](InputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, adjW, adjH,nGroup, noBias,wRegularizer,bRegularizer)
```
**Python:**
```python
m = SpatialFullConvolution(n_input_plane,n_output_plane,kw,kh,dw=1,dh=1,pad_w=0,pad_h=0,adj_w=0,adj_h=0,n_group=1,no_bias=False,init_method='default',wRegularizer=None,bRegularizer=None,)
```

SpatialFullConvolution is a module that applies a 2D full convolution over an input image. 

The input tensor in `forward(input)` is expected to be
either a 4D tensor (`batch x nInputPlane x height x width`) or a 3D tensor (`batch x height x width`). The convolution is performed on the last two dimensions. `adjW` and `adjH` are used to adjust the size of the output image. The size of output tensor of `forward` will be :
```
  output width  = (width  - 1) * dW - 2*padW + kW + adjW
  output height = (height - 1) * dH - 2*padH + kH + adjH
``` 

Note, scala API also accepts a table input with two tensors: `T(convInput, sizeTensor)` where `convInput` is the standard input tensor, and the size of `sizeTensor` is used to set the size of the output (will ignore the `adjW` and `adjH` values used to construct the module). Use `SpatialFullConvolution[Table, T](...)` instead of `SpatialFullConvolution[Tensor,T](...)`) for table input.
 
This module can also be used without a bias by setting parameter `noBias = true` while constructing the module.
 
Other frameworks may call this operation "In-network Upsampling", "Fractionally-strided convolution", "Backwards Convolution," "Deconvolution", or "Upconvolution."
 
Reference: Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015: 3431-3440.

**Scala example:**

Tensor Input example: 

```scala
scala> val input = Tensor[Double](1,1,3,3).randn()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,1,.,.) =
-0.038629937205745556   0.3184910844608199      -0.373807857057525
0.8239576087536957      -1.351254567086673      -1.4313258833988647
0.3927705082229262      -0.7216666620001434     1.1648889003942786

[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 1x1x3x3]

scala> val m = SpatialFullConvolution[Tensor[Double], Double](1, 2, 2, 2, 1, 1,0, 0, 0, 0, 1, false)
m: com.intel.analytics.bigdl.nn.SpatialFullConvolution[com.intel.analytics.bigdl.tensor.Tensor[Double],Double] = SpatialFullConvolution[1c6402e1](1 -> 2, 2 x 2, 1, 1, 0, 0, 0, 0)

scala> m.setInitMethod(weightInitMethod = BilinearFiller, biasInitMethod = Zeros)
res3: m.type = SpatialFullConvolution[1c6402e1](1 -> 2, 2 x 2, 1, 1, 0, 0, 0, 0)

scala> m.getParameters()
res4: (com.intel.analytics.bigdl.tensor.Tensor[Double], com.intel.analytics.bigdl.tensor.Tensor[Double]) =
(1.0
0.0
0.0
0.0
1.0
0.0
0.0
0.0
0.0
0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 10],0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 10])

scala> m.forward(input)
res7: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,1,.,.) =
-0.038629937205745556   0.3184910844608199      -0.373807857057525      0.0
0.8239576087536957      -1.351254567086673      -1.4313258833988647     0.0
0.3927705082229262      -0.7216666620001434     1.1648889003942786      0.0
0.0     0.0     0.0     0.0

(1,2,.,.) =
-0.038629937205745556   0.3184910844608199      -0.373807857057525      0.0
0.8239576087536957      -1.351254567086673      -1.4313258833988647     0.0
0.3927705082229262      -0.7216666620001434     1.1648889003942786      0.0
0.0     0.0     0.0     0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x4x4]

scala> val gradOut = Tensor[Double](1,2,4,4).fill(0.1)
gradOut: com.intel.analytics.bigdl.tensor.Tensor[Double] =
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

[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 1x2x4x4]

scala> m.backward(input,gradOut)
res8: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,1,.,.) =
0.2     0.2     0.2
0.2     0.2     0.2
0.2     0.2     0.2

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x3x3]

scala> m.getParameters()
res10: (com.intel.analytics.bigdl.tensor.Tensor[Double], com.intel.analytics.bigdl.tensor.Tensor[Double]) =
(1.0
0.0
0.0
0.0
1.0
0.0
0.0
0.0
0.0
0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 10],-0.12165768049172312
-0.12165768049172312
-0.12165768049172312
-0.12165768049172312
-0.12165768049172312
-0.12165768049172312
-0.12165768049172312
-0.12165768049172312
1.6
1.6
[com.intel.analytics.bigdl.tensor.DenseTensor of size 10])
	
```

Table input Example
```scala
scala> val input1 = Tensor[Double](1, 3, 3).randn()
input1: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,.,.) =
0.30435155168071043     1.3692609090335266      -0.8471308763085015
-0.6010343097036569     -0.16001902365874854    0.7600208146441209
0.09163405381781076     2.36238256006474        1.5839981410098911

[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 1x3x3]

scala> val input2 = Tensor[Double](3, 3).fill(2)
input2: com.intel.analytics.bigdl.tensor.Tensor[Double] =
2.0     2.0     2.0
2.0     2.0     2.0
2.0     2.0     2.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x3]

scala> import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.T

scala> val input = T(input1, input2)
input: com.intel.analytics.bigdl.utils.Table =
 {
        2: 2.0  2.0     2.0
           2.0  2.0     2.0
           2.0  2.0     2.0
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x3]
        1: (1,.,.) =
           0.30435155168071043  1.3692609090335266      -0.8471308763085015
           -0.6010343097036569  -0.16001902365874854    0.7600208146441209
           0.09163405381781076  2.36238256006474        1.5839981410098911

           [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 1x3x3]
 }
 
scala> val m = SpatialFullConvolution[Tensor[Double], Double](1, 2, 2, 2, 1, 1,0, 0, 0, 0, 1, false)
2017-06-22 01:37:09 INFO  ThreadPool$:79 - Set mkl threads to 1 on thread 1
m: com.intel.analytics.bigdl.nn.SpatialFullConvolution[com.intel.analytics.bigdl.tensor.Tensor[Double],Double] = SpatialFullConvolution[985189f4](1 -> 2, 2 x 2, 1, 1, 0, 0, 0, 0)

scala> m.forward(input)
res1: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.2602668232495975      -0.259424377873712      0.3402447511722818      0.6245106622771603
0.7626053209853888      1.100473402288004       -0.8007068522894367     0.5075845979676633
0.08234152211299228     -0.4329573387855421     -0.532326738356816      -0.3696785916677314
0.4244334729683819      1.3869784890191517      0.13534571019355257     -0.25005341061617825

(2,.,.) =
-0.08801024988174377    -0.04770658226342539    0.30212159978923975     -0.293649207253451
-0.14838633925904215    -0.7479004865288426     -0.37400557503044185    0.41727630735837096
0.10152105767196751     0.12475137015155185     0.29686019364250515     0.0575891353298002
-0.10507232658518488    -0.8254006905535733     -1.3951301984761595     -0.6392103148890831

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x4]
```

**Python example:**
```python
model = Sequential()
model.add(SpatialFullConvolution(1, 1, 1, 1).set_init_method(BilinearFiller(), Zeros()))
```
