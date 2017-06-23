## SpatialConvolution ##

**Scala:**
```scala
val m = SpatialConvolution[T](nInputPlane,nOutputPlane,kernelW,kernelH,strideW,strideH,padW,padH,nGroup,propagateBack,wRegularizer,bRegularizer,initWeight, initBias, initGradWeight, initGradBias)
```
**Python:**
```python
m = SpatialConvolution(n_input_plane,n_output_plane,kernel_w,kernel_h,stride_w=1,stride_h=1,pad_w=0,pad_h=0,n_group=1,propagate_back=True,init_method="default")
```

SpatialConvolution is a module that applies a 2D convolution over an input image.

The input tensor in `forward(input)` is expected to be
either a 4D tensor (`batch x nInputPlane x height x width`) or a 3D tensor (`batch x height x width`). The convolution is performed on the last two dimensions.

**Scala example:**
```scala
scala> val input = Tensor[Double](1,2,3,3).randn()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,1,.,.) =
-0.15036706177525752    0.6812241325457827      -0.6850390156337345
-0.1914930670115203     -0.5915171727695424     -0.6514810516405148
-0.6068519920134171     0.16450267576382982     -0.2486342371203115

(1,2,.,.) =
-0.9283595025163413     -0.02194934330100707    0.7781211906256997
-1.100526902178938      -0.5722917467649458     1.0299206067029267
1.2994160475191487      -0.5502152506035778     2.0080993032094896

[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 1x2x3x3]

scala> val m = SpatialConvolution[Double](2,1,2,2,1,1,0,0)
m: com.intel.analytics.bigdl.nn.SpatialConvolution[Double] = SpatialConvolution[9a2fb54a](2 -> 1, 2 x 2, 1, 1, 0, 0)

scala> m.setInitMethod(weightInitMethod = BilinearFiller, biasInitMethod = Zeros)
res41: m.type = SpatialConvolution[9a2fb54a](2 -> 1, 2 x 2, 1, 1, 0, 0)

scala> m.getParameters()
res43: (com.intel.analytics.bigdl.tensor.Tensor[Double], com.intel.analytics.bigdl.tensor.Tensor[Double]) =
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

scala> m.forward(input)
res44: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,1,.,.) =
-1.0787265642915989     0.6592747892447757
-1.2920199691904581     -1.163808919534488

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x2x2]

scala> val gradOut = Tensor[Double](1,1,2,2).fill(0.2)
gradOut: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,1,.,.) =
0.2     0.2
0.2     0.2

[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 1x1x2x2]

scala> m.backward(input,gradOut)
res45: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,1,.,.) =
0.2     0.2     0.0
0.2     0.2     0.0
0.0     0.0     0.0

(1,2,.,.) =
0.2     0.2     0.0
0.2     0.2     0.0
0.0     0.0     0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x3x3]

scala> m.getParameters()
res47: (com.intel.analytics.bigdl.tensor.Tensor[Double], com.intel.analytics.bigdl.tensor.Tensor[Double]) =
(1.0
0.0
0.0
0.0
1.0
0.0
0.0
0.0
0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 9],-0.050430633802107494
-0.2493626214996018
-0.24507191120613
-0.2654259571533078
-0.5246254989522464
0.24276014145253474
-0.18472357040566256
0.38310258250877854
0.8
[com.intel.analytics.bigdl.tensor.DenseTensor of size 9])
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
grad_in = m.backward(input,grad_out)
print "grad input of m is :",grad_in
```
produces output:
```python
input is : [[[[ 0.10278214  0.12979619  0.72584278]
   [ 0.26983861  0.14588091  0.44783144]
   [ 0.22259862  0.54787749  0.34462945]]

  [[ 0.9473658   0.83794153  0.6395567 ]
   [ 0.86116576  0.87821085  0.29496161]
   [ 0.00870506  0.56188944  0.21540011]]

  [[ 0.98895581  0.2122084   0.22285177]
   [ 0.79510004  0.83722934  0.87748588]
   [ 0.51133915  0.62574427  0.95832323]]]]
creating: createSpatialConvolution
output m is : [array([[[[ 0.26038894,  0.06098987],
         [ 0.32970002,  0.23258424]]]], dtype=float32)]
grad input of m is : [array([[[[ -1.51922330e-01,  -1.00127935e-01,  -6.00575609e-03],
         [ -1.74155504e-01,  -2.46597916e-01,  -5.84947988e-02],
         [ -3.23428065e-02,  -1.07444011e-01,  -7.31263086e-02]],

        [[  2.03749567e-01,   2.29789000e-02,   6.26069377e-04],
         [ -1.06653631e-01,   2.83021927e-02,   1.02799386e-04],
         [ -3.59356217e-02,  -1.02768913e-01,  -4.19370718e-02]],

        [[  1.46538138e-01,   1.09280780e-01,   6.64058886e-03],
         [  2.37367854e-01,   3.00488621e-02,   4.90537882e-02],
         [  4.73713949e-02,   9.17033181e-02,  -4.83076982e-02]]]], dtype=float32)]
```