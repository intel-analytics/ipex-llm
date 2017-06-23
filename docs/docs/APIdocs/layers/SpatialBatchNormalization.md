## SpatialBatchNormalization ##

**Scala:**
```scala
val module = SpatialBatchNormalization(nOutput, eps, momentum, affine,
                                           initWeight, initBias, initGradWeight, initGradBias)
```
**Python:**
```python
module = SpatialBatchNormalization(nOutput, eps, momentum, affine,
                                       initWeight, initBias, initGradWeight, initGradBias)
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
`nOutput` output feature map number

`eps` avoid divide zero

`momentum` momentum for weight update

`affine` affine operation on output or not

`initWeight` initial weight tensor

`initBias`  initial bias tensor

`initGradWeight` initial gradient weight 

`initGradBias` initial gradient bias
 
 
**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

val layer = new SpatialBatchNormalization[Float](3, 1e-3)
val input = Tensor[Float](2, 3, 2, 2).randn()
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