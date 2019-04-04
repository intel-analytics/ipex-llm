## BatchNormalization ##

**Scala:**
```scala
val bn = BatchNormalization(nOutput, eps, momentum, affine)
```
**Python:**
```python
bn = BatchNormalization(n_output, eps, momentum, affine)
```

This layer implements Batch Normalization as described in the paper:
[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
by Sergey Ioffe, Christian Szegedy

This implementation is useful for inputs NOT coming from convolution layers. For convolution layers, use nn.SpatialBatchNormalization.

The operation implemented is:

```
              ( x - mean(x) )
      y =  -------------------- * gamma + beta
              standard-deviation(x)
```
where gamma and beta are learnable parameters.The learning of gamma and beta is optional.

Parameters:

* `nOutput` feature map number
* `eps` avoid divide zero. Default: 1e-5
* `momentum` momentum for weight update. Default: 0.1
* `affine` affine operation on output or not. Default: true

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val bn = BatchNormalization(2)
val input = Tensor(T(
             T(1.0f, 2.0f),
             T(3.0f, 6.0f))
            )
val gradOutput = Tensor(T(
             T(1.0f, 2.0f),
             T(3.0f, 6.0f))
)
val output = bn.forward(input)
val gradient = bn.backward(input, gradOutput)
-> print(output) 
# There's random factor. An output could be
-0.46433213     -0.2762179      
0.46433213      0.2762179       
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
-> print(gradient)
# There's random factor. An output could be
-4.649627E-6    -6.585548E-7    
4.649627E-6     6.585548E-7     
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
bn = BatchNormalization(2)
input = np.array([
  [1.0, 2.0],
  [3.0, 6.0]
])
grad_output = np.array([
           [2.0, 3.0],
           [4.0, 5.0]
         ])
output = bn.forward(input)
gradient = bn.backward(input, grad_output)
-> print output
# There's random factor. An output could be
[[-0.99583918 -0.13030811]
 [ 0.99583918  0.13030811]]
-> print gradient
# There's random factor. An output could be
[[ -9.97191637e-06  -1.55339364e-07]
 [  9.97191637e-06   1.55339364e-07]]
```
---
## SpatialBatchNormalization ##

**Scala:**
```scala
val module = SpatialBatchNormalization(nOutput, eps=1e-5, momentum=0.1, affine=true,
                                           initWeight=null, initBias=null, initGradWeight=null, initGradBias=null)
```
**Python:**
```python
module = SpatialBatchNormalization(nOutput, eps=1e-5, momentum=0.1, affine=True)

```

This file implements Batch Normalization as described in the paper:
"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
by Sergey Ioffe, Christian Szegedy.

This implementation is useful for inputs coming from convolution layers.
For non-convolutional layers, see `BatchNormalization`
The operation implemented is:
``` 
        ( x - mean(x) )
  y = -------------------- * gamma + beta
       standard-deviation(x)
 
  where gamma and beta are learnable parameters.
  The learning of gamma and beta is optional.
```

+ `nOutput` output feature map number
+ `eps` avoid divide zero
+ `momentum` momentum for weight update
+ `affine` affine operation on output or not
+ `initWeight` initial weight tensor
+ `initBias`  initial bias tensor
+ `initGradWeight` initial gradient weight 
+ `initGradBias` initial gradient bias
+ `data_format` a string value (or DataFormat Object in Scala) of "NHWC" or "NCHW" to specify the input data format of this layer. In "NHWC" format
                        data is stored in the order of \[batch_size, height, width, channels\], in "NCHW" format data is stored
                        in the order of \[batch_size, channels, height, width\].
 
 
**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val layer = SpatialBatchNormalization(3, 1e-3)
val input = Tensor(2, 3, 2, 2).randn()
> print(layer.forward(input))
(1,1,.,.) =
-0.21939678	-0.64394164	
-0.03280549	0.13889995	

(1,2,.,.) =
0.48519397	0.40222475	
-0.9339038	0.4131121	

(1,3,.,.) =
0.39790314	-0.040012743	
-0.009540742	0.21598668	

(2,1,.,.) =
0.32008895	-0.23125978	
0.4053611	0.26305377	

(2,2,.,.) =
-0.3810518	-0.34581286	
0.14797378	0.21226381	

(2,3,.,.) =
0.2558251	-0.2211882	
-0.59388477	-0.00508846	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x2x2]
```

**Python example:**
```python
from bigdl.nn.layer import *

layer = SpatialBatchNormalization(3, 1e-3)
input = np.random.rand(2,3,2,2)
>layer.forward(input)
array([[[[  5.70826093e-03,   9.06338100e-05],
         [ -3.49177676e-03,   1.10401707e-02]],

        [[  1.80168569e-01,  -8.87815133e-02],
         [  2.11335659e-01,   2.11817324e-01]],

        [[ -1.02916014e+00,   4.02444333e-01],
         [ -1.72453150e-01,   5.31806648e-01]]],


       [[[ -3.46255396e-03,  -1.37512591e-02],
         [  3.84721952e-03,   1.93112865e-05]],

        [[  4.65962708e-01,  -5.29752195e-01],
         [ -2.28064612e-01,  -2.22685724e-01]],

        [[  8.49217057e-01,  -9.03094828e-01],
         [  8.56826544e-01,  -5.35586655e-01]]]], dtype=float32)
```
---
## SpatialCrossMapLRN ##

**Scala:**
```scala
val spatialCrossMapLRN = SpatialCrossMapLRN(size = 5, alpha  = 1.0, beta = 0.75, k = 1.0)
```
**Python:**
```python
spatialCrossMapLRN = SpatialCrossMapLRN(size=5, alpha=1.0, beta=0.75, k=1.0)
```

SpatialCrossMapLRN applies Spatial Local Response Normalization between different feature maps

```
                             x_f
  y_f =  -------------------------------------------------
          (k+(alpha/size)* sum_{l=l1 to l2} (x_l^2^))^beta^
          
where  l1 corresponds to `max(0,f-ceil(size/2))` and l2 to `min(F, f-ceil(size/2) + size)`, `F` is the number  of feature maps       
```

+ `size`  the number of channels to sum over
+ `alpha`  the scaling parameter
+ `beta`   the exponent
+ `k` a constant
+ `data_format` a string value (or DataFormat Object in Scala) of "NHWC" or "NCHW" to specify the input data format of this layer. In "NHWC" format
                        data is stored in the order of \[batch_size, height, width, channels\], in "NCHW" format data is stored
                        in the order of \[batch_size, channels, height, width\]

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val spatialCrossMapLRN = SpatialCrossMapLRN(5, 0.01, 0.75, 1.0)

val input = Tensor(2, 2, 2, 2).rand()

> print(input)
(1,1,.,.) =
0.42596373	0.20075735	
0.10307904	0.7486494	

(1,2,.,.) =
0.9887414	0.3554662	
0.6291069	0.53952795	

(2,1,.,.) =
0.41220918	0.5463298	
0.40766734	0.08064394	

(2,2,.,.) =
0.58255607	0.027811589	
0.47811228	0.3082057	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x2x2]

> print(spatialCrossMapLRN.forward(input))
(1,1,.,.) =
0.42522463	0.20070718	
0.10301625	0.74769455	

(1,2,.,.) =
0.98702586	0.35537735	
0.6287237	0.5388398	

(2,1,.,.) =
0.41189456	0.5460847	
0.4074261	0.08063166	

(2,2,.,.) =
0.5821114	0.02779911	
0.47782937	0.3081588	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2]

```

**Python example:**
```python
from bigdl.nn.layer import *
spatialCrossMapLRN = SpatialCrossMapLRN(5, 0.01, 0.75, 1.0)
> spatialCrossMapLRN.forward(np.array([[[[1, 2],[3, 4]],[[5, 6],[7, 8]]],[[[9, 10],[11, 12]],[[13, 14],[15, 16]]]]))
[array([[[[  0.96269381,   1.88782692],
         [  2.76295042,   3.57862759]],

        [[  4.81346893,   5.66348076],
         [  6.44688463,   7.15725517]]],


       [[[  6.6400919 ,   7.05574226],
         [  7.41468   ,   7.72194815]],

        [[  9.59124374,   9.87803936],
         [ 10.11092758,  10.29593086]]]], dtype=float32)]

     
```
---
## SpatialWithinChannelLRN ##

**Scala:**
```scala
val spatialWithinChannelLRN = SpatialWithinChannelLRN(size = 5, alpha  = 1.0, beta = 0.75)
```
**Python:**
```python
spatialWithinChannelLRN = SpatialWithinChannelLRN(size=5, alpha=1.0, beta=0.75)
```

SpatialWithinChannelLRN performs a kind of “lateral inhibition”
by normalizing over local input regions. the local regions extend spatially,
in separate channels (i.e., they have shape 1 x size x size).

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val spatialWithinChannelLRN = SpatialWithinChannelLRN(5, 0.01, 0.75)

val input = Tensor(2, 2, 2, 2).rand()

> print(input)
(1,1,.,.) =
0.8658837       0.1297312
0.7559588       0.039047405

(1,2,.,.) =
0.79211944      0.84445393
0.8854509       0.6596644

(2,1,.,.) =
0.96907943      0.7036902
0.90358996      0.5719087

(2,2,.,.) =
0.52309155      0.8838519
0.44981572      0.40950212

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x2x2]

> print(spatialWithinChannelLRN.forward(input))
(1,1,.,.) =
0.8655359       0.12967908      
0.75565517      0.03903172      

(1,2,.,.) =
0.7915117       0.843806        
0.8847715       0.6591583       

(2,1,.,.) =
0.9683307       0.70314646      
0.9028918       0.5714668       

(2,2,.,.) =
0.52286804      0.8834743       
0.44962353      0.40932715      

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2]

```

**Python example:**
```python
from bigdl.nn.layer import *
spatialWithinChannelLRN = SpatialWithinChannelLRN(5, 0.01, 0.75)
> spatialWithinChannelLRN.forward(np.array([[[[1, 2],[3, 4]],[[5, 6],[7, 8]]],[[[9, 10],[11, 12]],[[13, 14],[15, 16]]]]))
array([[[[  0.99109352,   1.98218703],
         [  2.97328043,   3.96437407]],

        [[  4.75394297,   5.70473146],
         [  6.65551996,   7.60630846]]],


       [[[  7.95743227,   8.84159184],
         [  9.72575092,  10.60991001]],

        [[ 10.44729614,  11.2509346 ],
         [ 12.05457211,  12.85821056]]]], dtype=float32)

     
```
---
## Normalize ##

**Scala:**
```scala
val module = Normalize(p,eps=1e-10)
```
**Python:**
```python
module = Normalize(p,eps=1e-10,bigdl_type="float")
```

Normalizes the input Tensor to have unit L_p norm. The smoothing parameter eps prevents
division by zero when the input contains all zero elements (default = 1e-10).
The input can be 1d, 2d or 4d. If the input is 4d, it should follow the format (n, c, h, w) where n is the batch number,
c is the channel number, h is the height and w is the width

 *  `p` L_p norm
 *  `eps` smoothing parameter

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val module = Normalize(2.0,eps=1e-10)
val input = Tensor(2,3).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.7075603       0.084298864     0.91339105
0.22373432      0.8704987       0.6936567
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

module.forward(input)
res8: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.6107763       0.072768        0.7884524
0.19706465      0.76673317      0.61097115
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
module = Normalize(2.0,eps=1e-10,bigdl_type="float")
input = np.array([[1, 2, 3],[4, 5, 6]])
module.forward(input)
[array([
[ 0.26726124,  0.53452247,  0.80178368],
[ 0.45584232,  0.56980288,  0.68376344]], dtype=float32)]
```

## SpatialDivisiveNormalization ##

**Scala:**
```scala
val layer = SpatialDivisiveNormalization()
```
**Python:**
```python
layer = SpatialDivisiveNormalization()
```

Applies a spatial division operation on a series of 2D inputs using kernel for
computing the weighted average in a neighborhood. The neighborhood is defined for
a local spatial region that is the size as kernel and across all features. For
an input image, since there is only one feature, the region is only spatial. For
an RGB image, the weighted average is taken over RGB channels and a spatial region.

If the kernel is 1D, then it will be used for constructing and separable 2D kernel.
The operations will be much more efficient in this case.

The kernel is generally chosen as a gaussian when it is believed that the correlation
of two pixel locations decrease with increasing distance. On the feature dimension,
a uniform average is used since the weighting across features is not known.

**Scala example:**
```scala

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val layer = SpatialDivisiveNormalization()
val input = Tensor(1, 5, 5).rand
val gradOutput = Tensor(1, 5, 5).rand

val output = layer.forward(input)
val gradInput = layer.backward(input, gradOutput)

> println(input)
res19: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.4022106       0.6872489       0.9712838       0.7769542       0.771034
0.97930336      0.61022973      0.65092266      0.9507807       0.3158211
0.12607759      0.320569        0.9267993       0.47579524      0.63989824
0.713135        0.30836385      0.009723447     0.67723924      0.24405171
0.51036286      0.115807846     0.123513035     0.28398398      0.271164

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5x5]

> println(output)
res20: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.37849638      0.6467289       0.91401714      0.73114514      0.725574
0.9215639       0.57425076      0.6125444       0.89472294      0.29720038
0.11864409      0.30166835      0.8721555       0.4477425       0.60217
0.67108876      0.2901828       0.009150156     0.6373094       0.2296625
0.480272        0.10897984      0.11623074      0.26724035      0.25517625

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5x5]

> println(gradInput)
res21: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.09343022     -0.25612304     0.25756648      -0.66132677     -0.44575396
0.052990615     0.7899354       0.27205157      0.028260134     0.23150417
-0.115425855    0.21133065      0.53093016      -0.36421964     -0.102551565
0.7222408       0.46287358      0.0010696054    0.26336592      -0.050598443
0.03733714      0.2775169       -0.21430963     0.3175013       0.6600435

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5x5]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

layer = SpatialDivisiveNormalization()
input = np.random.uniform(0, 1, (1, 5, 5)).astype("float32")
gradOutput = np.random.uniform(0, 1, (1, 5, 5)).astype("float32")

output = layer.forward(input)
gradInput = layer.backward(input, gradOutput)

> output
[array([[[ 0.30657911,  0.75221181,  0.2318386 ,  0.84053135,  0.24818985],
         [ 0.32852787,  0.43504578,  0.0219258 ,  0.47856906,  0.31112722],
         [ 0.12381417,  0.61807972,  0.90043157,  0.57342309,  0.65450585],
         [ 0.00401461,  0.33700454,  0.79859954,  0.64382601,  0.51768768],
         [ 0.38087726,  0.8963666 ,  0.7982524 ,  0.78525543,  0.09658573]]], dtype=float32)]
> gradInput
[array([[[ 0.08059166, -0.4616771 ,  0.11626807,  0.30253756,  0.7333734 ],
         [ 0.2633073 , -0.01641282,  0.40653706,  0.07766753, -0.0237394 ],
         [ 0.10733987,  0.23385212, -0.3291783 , -0.12808481,  0.4035565 ],
         [ 0.56126803,  0.49945205, -0.40531909, -0.18559581,  0.27156472],
         [ 0.28016835,  0.03791744, -0.17803842, -0.27817759,  0.42473239]]], dtype=float32)]
```

---
## SpatialSubtractiveNormalization ##

**Scala:**
```scala
val spatialSubtractiveNormalization = SpatialSubtractiveNormalization(nInputPlane = 1, kernel = null)
```
**Python:**
```python
spatialSubtractiveNormalization = SpatialSubtractiveNormalization(n_input_plane=1, kernel=None)
```

SpatialSubtractiveNormalization applies a spatial subtraction operation on a series of 2D inputs using kernel for computing the weighted average in a neighborhood.The neighborhood is defined for a local spatial region that is the size as kernel and across all features. For an input image, since there is only one feature, the region is only spatial. For an RGB image, the weighted average is taken over RGB channels and a spatial region.

If the kernel is 1D, then it will be used for constructing and separable 2D kernel.
The operations will be much more efficient in this case.
 
The kernel is generally chosen as a gaussian when it is believed that the correlation
of two pixel locations decrease with increasing distance. On the feature dimension,
a uniform average is used since the weighting across features is not known.

+ `nInputPlane`  number of input plane, default is 1.
+ `kernel` kernel tensor, default is a 9 x 9 tensor.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val kernel = Tensor(3, 3).rand()

> print(kernel)
0.56141114	0.76815456	0.29409808	
0.3599753	0.17142025	0.5243272	
0.62450963	0.28084084	0.17154165	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]


val spatialSubtractiveNormalization = SpatialSubtractiveNormalization(1, kernel)

val input = Tensor(1, 1, 1, 5).rand()

> print(input)
(1,1,.,.) =
0.122356184	0.44442436	0.6394927	0.9349956	0.8226007	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x1x1x5]

> print(spatialSubtractiveNormalization.forward(input))
(1,1,.,.) =
-0.2427161	0.012936085	-0.08024883	0.15658027	-0.07613802	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x1x5]


```
---
**Python example:**
```python
from bigdl.nn.layer import *
kernel=np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
spatialSubtractiveNormalization = SpatialSubtractiveNormalization(1, kernel)
>  spatialSubtractiveNormalization.forward(np.array([[[[1, 2, 3, 4, 5]]]]))
[array([[[[ 0.,  0.,  0.,  0.,  0.]]]], dtype=float32)]

     
```


