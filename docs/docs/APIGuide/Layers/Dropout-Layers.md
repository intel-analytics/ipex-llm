## Dropout ##

**Scala:**
```scala
val module = Dropout(
  initP = 0.5,
  inplace = false,
  scale = true)
```
**Python:**
```python
module = Dropout(
  init_p=0.5,
  inplace=False,
  scale=True)
```

Dropout masks(set to zero) parts of input using a Bernoulli distribution.
Each input element has a probability `initP` of being dropped. If `scale` is
true(true by default), the outputs are scaled by a factor of `1/(1-initP)` during training.
During evaluating, output is the same as input.

It has been proven an effective approach for regularization and preventing
co-adaptation of feature detectors. For more details, please see
[Improving neural networks by preventing co-adaptation of feature detectors]
(https://arxiv.org/abs/1207.0580)

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val module = Dropout()
val x = Tensor.range(1, 8, 1).resize(2, 4)

println(module.forward(x))
println(module.backward(x, x.clone().mul(0.5f))) // backward drops out the gradients at the same location.
```
Output is
```
com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.0     4.0     6.0     0.0
10.0    12.0    0.0     16.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]

com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.0    2.0    3.0    0.0
5.0    6.0    0.0    8.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Dropout()
x = np.arange(1, 9, 1).reshape(2, 4)

print(module.forward(x))
print(module.backward(x, x.copy() * 0.5)) # backward drops out the gradients at the same location.
```
Output is
```
[array([[ 0.,  4.,  6.,  0.],
       [ 0.,  0.,  0.,  0.]], dtype=float32)]
       
[array([[ 0.,  2.,  3.,  0.],
       [ 0.,  0.,  0.,  0.]], dtype=float32)]
```


## GaussianDropout

**Scala:**
```scala
val module = GaussianDropout(rate)
```
**Python:**
```python
module = GaussianDropout(rate)
```

Apply multiplicative 1-centered Gaussian noise.
As it is a regularization layer, it is only active at training time.

* `rate` is drop probability (as with `Dropout`).

Reference: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.nn._

val layer = GaussianDropout(0.5)
layer.training()
val input = Tensor(T(T(1.0,1.0,1.0),T(1.0,1.0,1.0)))
val output = layer.forward(input)
val gradout = Tensor(T(T(1.0,1.0,1.0),T(1.0,1.0,1.0)))
val gradin = layer.backward(input,gradout)
println(output)
println(gradin)
layer.evaluate()
val output2 = layer.forward(input)
println(output2)
```
Output is
```
1.7464018	2.9785068	0.053465042
0.6711602	2.7494855	0.13988598
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
0.86814594	1.9510038	1.5220107
1.2875593	0.10056248	2.5501933
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
1.0	1.0	1.0
1.0	1.0	1.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
layer = GaussianDropout(0.5) # Try to create a Linear layer

#training mode
layer.training()
inp=np.ones([2,1])
outp = layer.forward(inp)

gradoutp = np.ones([2,1])
gradinp = layer.backward(inp,gradoutp)
print "training:forward=",outp
print "trainig:backward=",gradinp

#evaluation mode
layer.evaluate()
print "evaluate:forward=",layer.forward(inp)

```
Output is
```
creating: createGaussianDropout
training:forward= [[ 0.80695641]
 [ 1.82794702]]
trainig:backward= [[ 0.1289842 ]
 [ 1.22549391]]
evaluate:forward= [[ 1.]
 [ 1.]]

```

## GaussianNoise

**Scala:**
```scala
val module = GaussianNoise(stddev)
```
**Python:**
```python
module = GaussianNoise(stddev)
```

Apply additive zero-centered Gaussian noise. This is useful to mitigate overfitting (you could see it as a form of random data augmentation).
Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs.

As it is a regularization layer, it is only active at training time.

* `stddev` is the standard deviation of the noise distribution.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.nn._

val layer = GaussianNoise(0.2)
layer.training()
val input = Tensor(T(T(1.0,1.0,1.0),T(1.0,1.0,1.0)))
val output = layer.forward(input)
val gradout = Tensor(T(T(1.0,1.0,1.0),T(1.0,1.0,1.0)))
val gradin = layer.backward(input,gradout)
layer.evaluate()
println(layer.forward(input))
```
```
1.0	1.0	1.0
1.0	1.0	1.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]

```

**Python example:**
```python
layer = GaussianNoise(0.5) 

#training mode
layer.training()
inp=np.ones([2,1])
outp = layer.forward(inp)

gradoutp = np.ones([2,1])
gradinp = layer.backward(inp,gradoutp)
print "training:forward=",outp
print "trainig:backward=",gradinp

#evaluation mode
layer.evaluate()
print "evaluate:forward=",layer.forward(inp)

```
Output is
```
creating: createGaussianNoise
training:forward= [[ 0.99984151]
 [ 1.11269045]]
trainig:backward= [[ 1.]
 [ 1.]]
evaluate:forward= [[ 1.]
 [ 1.]]
```

## SpatialDropout1D ##

**Scala:**
```scala
val module = SpatialDropout1D(initP = 0.5)
```
**Python:**
```python
module = SpatialDropout1D(
  init_p=0.5)
```

This version performs the same function as Dropout, however it drops
   entire 1D feature maps instead of individual elements. If adjacent frames
   within feature maps are strongly correlated (as is normally the case in
   early convolution layers) then regular dropout will not regularize the
   activations and will otherwise just result in an effective learning rate
   decrease. In this case, SpatialDropout1D will help promote independence
   between feature maps and should be used instead.
 
 * `initP` the probability p

**Scala example:**
```scala
    import com.intel.analytics.bigdl.tensor.Tensor
    import com.intel.analytics.bigdl.nn._
    import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

    val module = SpatialDropout1D[Double](0.7)
    val input = Tensor[Double](3, 4, 5)
    val seed = 100

    input.rand()

    RNG.setSeed(seed)
    val output = module.forward(input)
    > println(output)
    (1,.,.) =
    0.0	0.0	0.8925298328977078	0.0	0.0	
    0.0	0.0	0.8951127317268401	0.0	0.0	
    0.0	0.0	0.425491401925683	0.0	0.0	
    0.0	0.0	0.31143878563307226	0.0	0.0	
    
    (2,.,.) =
    0.0	0.0	0.06833203043788671	0.5629170550964773	0.49213682673871517	
    0.0	0.0	0.5263364950660616	0.5756838673260063	0.060498124454170465	
    0.0	0.0	0.8886410375125706	0.539079936221242	0.4533065736759454	
    0.0	0.0	0.8942249100655317	0.5489360291976482	0.05561425327323377	
    
    (3,.,.) =
    0.007322707446292043	0.07132467231713235	0.0	0.3080112475436181	0.0	
    0.8506345122586936	0.383204679004848	0.0	0.9952241901773959	0.0	
    0.6507184051442891	0.20175716653466225	0.0	0.28786351275630295	0.0	
    0.19677149993367493	0.3048216907773167	0.0	0.5715036438778043	0.0	
    
    [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4x5]
    
    val gradInput = module.backward(input, input.clone().fill(1))
    > println(gradInput)
    (1,.,.) =
    0.0	0.0	1.0	0.0	0.0	
    0.0	0.0	1.0	0.0	0.0	
    0.0	0.0	1.0	0.0	0.0	
    0.0	0.0	1.0	0.0	0.0	
    
    (2,.,.) =
    0.0	0.0	1.0	1.0	1.0	
    0.0	0.0	1.0	1.0	1.0	
    0.0	0.0	1.0	1.0	1.0	
    0.0	0.0	1.0	1.0	1.0	
    
    (3,.,.) =
    1.0	1.0	0.0	1.0	0.0	
    1.0	1.0	0.0	1.0	0.0	
    1.0	1.0	0.0	1.0	0.0	
    1.0	1.0	0.0	1.0	0.0	
    
    [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4x5]
```


**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = SpatialDropout1D(0.7)
x = np.arange(3, 4, 5)

print(module.forward(x))
print(module.backward(x, x.copy() * 0.5)) # backward drops out the gradients at the same location.
```
Output is
```
 [[[0.0	0.0	0.8925298328977078	0.0	0.0]	
    [0.0	0.0	0.8951127317268401	0.0	0.0]	
    [0.0	0.0	0.425491401925683	0.0	0.0]	
    [0.0	0.0	0.31143878563307226	0.0	0.0]]	
    
    [0.0	0.0	0.06833203043788671	0.5629170550964773	0.49213682673871517 ]	
    [0.0	0.0	0.5263364950660616	0.5756838673260063	0.060498124454170465 ]	
    [0.0	0.0	0.8886410375125706	0.539079936221242	0.4533065736759454 ]	
    [0.0	0.0	0.8942249100655317	0.5489360291976482	0.05561425327323377	]]
    
    [0.007322707446292043	0.07132467231713235	0.0	0.3080112475436181	0.0 ]	
    [0.8506345122586936	0.383204679004848	0.0	0.9952241901773959	0.0 ]	
    [0.6507184051442891	0.20175716653466225	0.0	0.28786351275630295	0.0 ]	
    [0.19677149993367493	0.3048216907773167	0.0	0.5715036438778043	0.0]]]
    
    
     [[[0.0	0.0	1.0	0.0	0.0]
        [0.0 0.0 1.0 0.0 0.0]	
        [0.0 0.0 1.0 0.0 0.0]	
        [0.0 0.0 1.0 0.0 0.0]]
        
       [[0.0 0.0 1.0 1.0 1.0]	
        [0.0 0.0 1.0 1.0 1.0]	
        [0.0 0.0 1.0 1.0 1.0]	
        [0.0 0.0 1.0 1.0 1.0]]
        
       [[1.0 1.0 0.0 1.0 0.0]	
        [1.0 1.0 0.0 1.0 0.0]	
        [1.0 1.0 0.0 1.0 0.0]	
        [1.0 1.0 0.0 1.0 0.0]]]
  
```

## SpatialDropout2D ##

**Scala:**
```scala
val module = SpatialDropout2D(initP = 0.5, format = DataFormat.NCHW)
```
**Python:**
```python
module = SpatialDropout2D(
  init_p=0.5, data_format="NCHW")
```

This version performs the same function as Dropout, however it drops
 entire 2D feature maps instead of individual elements. If adjacent pixels
 within feature maps are strongly correlated (as is normally the case in
 early convolution layers) then regular dropout will not regularize the
 activations and will otherwise just result in an effective learning rate
 decrease. In this case, SpatialDropout2D will help promote independence
 between feature maps and should be used instead.
 
 * param initP the probability p
 * param format  'NCHW' or 'NHWC'.
            In 'NCHW' mode, the channels dimension (the depth)
            is at index 1, in 'NHWC' mode is it at index 4.
 

**Scala example:**
```scala
    import com.intel.analytics.bigdl.tensor.Tensor
    import com.intel.analytics.bigdl.nn._
    import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
    
    val module = SpatialDropout2D[Double](0.7)
    val input = Tensor[Double](2, 3, 4, 5)
    val seed = 100

    input.rand()

    RNG.setSeed(seed)
    val output = module.forward(input)
    > println(output)
    (1,1,.,.) =
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    
    (1,2,.,.) =
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    
    (1,3,.,.) =
    0.9125777170993388	0.828888057731092	0.3860199467744678	0.4881938952021301	0.3932550342287868	
    0.3380460755433887	0.32206087466329336	0.9833535915240645	0.7536576387938112	0.6055934554897249	
    0.34218871919438243	0.045394203858450055	0.03498578444123268	0.6890419721603394	0.12134534679353237	
    0.3766667563468218	0.8550574257969856	0.16245933924801648	0.8359398010652512	0.9934550793841481	
    
    (2,1,.,.) =
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    
    (2,2,.,.) =
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    
    (2,3,.,.) =
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    
    [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4x5]

    
    val gradInput = module.backward(input, input.clone().fill(1))
    > println(gradInput)
    (1,1,.,.) =
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    
    (1,2,.,.) =
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    
    (1,3,.,.) =
    1.0	1.0	1.0	1.0	1.0	
    1.0	1.0	1.0	1.0	1.0	
    1.0	1.0	1.0	1.0	1.0	
    1.0	1.0	1.0	1.0	1.0	
    
    (2,1,.,.) =
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    
    (2,2,.,.) =
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    
    (2,3,.,.) =
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    0.0	0.0	0.0	0.0	0.0	
    
    [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4x5]

```


**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = SpatialDropout1D(0.7)
x = np.arange(3, 4, 5)

print(module.forward(x))
print(module.backward(x, x.copy() * 0.5)) # backward drops out the gradients at the same location.
```
Output is
```
output:
[[[0.0	0.0	0.0	0.0	0.0]
    [0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]]
    
   [[0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]]
    
   [[0.9125777170993388	0.828888057731092	0.3860199467744678	0.4881938952021301	0.3932550342287868]	
    [0.3380460755433887	0.32206087466329336	0.9833535915240645	0.7536576387938112	0.6055934554897249]
    [0.34218871919438243	0.045394203858450055	0.03498578444123268	0.6890419721603394	0.12134534679353237]
    [0.3766667563468218	0.8550574257969856	0.16245933924801648	0.8359398010652512	0.9934550793841481]
    
   [[0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]]
    
   [[0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]]
    
   [[0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]	
    [0.0	0.0	0.0	0.0	0.0]]]	


gradInput:
 [[[0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]]
     
    [[0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]]
     
    [[1.0	1.0	1.0	1.0	1.0]	
     [1.0	1.0	1.0	1.0	1.0]	
     [1.0	1.0	1.0	1.0	1.0]	
     [1.0	1.0	1.0	1.0	1.0]]
     
    [[0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]]
     
    [[0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]]
     
    [[0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]	
     [0.0	0.0	0.0	0.0	0.0]]]
```



## SpatialDropout3D ##

**Scala:**
```scala
val module = SpatialDropout3D(initP = 0.5, format = DataFormat.NCHW)
```
**Python:**
```python
module = SpatialDropout3D(
  init_p=0.5, data_format="NCHW")
```

 This version performs the same function as Dropout, however it drops
 entire 3D feature maps instead of individual elements. If adjacent voxels
 within feature maps are strongly correlated (as is normally the case in
 early convolution layers) then regular dropout will not regularize the
 activations and will otherwise just result in an effective learning rate
 decrease. In this case, SpatialDropout3D will help promote independence
 between feature maps and should be used instead.
 
 * `initP` the probability p
 * `format`  'NCHW' or 'NHWC'.
               In 'NCHW' mode, the channels dimension (the depth)
               is at index 1, in 'NHWC' mode is it at index 4.
```