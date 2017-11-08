## Scale ##


**Scala:**

```scala
val m = Scale(Array(2, 1))
```
**Python:**
```python
m = scale = Scale([2, 1])
```

Scale is the combination of cmul and cadd. `Scale(size).forward(input) == CAdd(size).forward(CMul(size).forward(input))`
Computes the elementwise product of input and weight, with the shape of the weight "expand" to
match the shape of the input.Similarly, perform a expand cdd bias and perform an elementwise add.
`output = input .* weight .+ bias (element wise)`


**Scala example:**
```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val input = Tensor(2, 3).fill(1f)
println("input:")
println(input)
val scale = Scale(Array(2, 1))
val weight = Tensor(2, 1).fill(2f)
val bias = Tensor(2, 1).fill(3f)
scale.setWeightsBias(Array(weight, bias))
println("Weight:")
println(weight)
println("bias:")
println(bias)
println("output:")
print(scale.forward(input))
```
```
input:
1.0	1.0	1.0	
1.0	1.0	1.0	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
Weight:
2.0	
2.0	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1]
bias:
3.0	
3.0	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1]
output:
5.0	5.0	5.0	
5.0	5.0	5.0	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**

``` python
import numpy as np
from bigdl.nn.layer import *
input = np.ones([2, 3])
print("input:")
print(input)
scale = Scale([2, 1])
weight = np.full([2, 1], 2)
bias = np.full([2, 1], 3)
print("weight: ")
print(weight)
print("bias: ")
print(bias)
scale.set_weights([weight, bias])
print("output: ")
print(scale.forward(input))

```
```
input:
[[ 1.  1.  1.]
 [ 1.  1.  1.]]
creating: createScale
weight: 
[[2]
 [2]]
bias: 
[[3]
 [3]]
output: 
[[ 5.  5.  5.]
 [ 5.  5.  5.]]
```


---
## Min ##

**Scala:**
```scala
val min = Min(dim, numInputDims)
```
**Python:**
```python
min = Min(dim, num_input_dims)
```

Applies a min operation over dimension `dim`.

Parameters:
* `dim` A integer. The dimension to min along.
* `numInputDims` An optional integer indicating the number of input dimensions.
 

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val min = Min(2)
val input = Tensor(T(
 T(1.0f, 2.0f),
 T(3.0f, 4.0f))
)
val gradOutput = Tensor(T(
 1.0f,
 1.0f
))
val output = min.forward(input)
val gradient = min.backward(input, gradOutput)
-> print(output)
1.0
3.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2]

-> print(gradient)
1.0     0.0     
1.0     0.0     
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
min = Min(2)
input = np.array([
  [1.0, 2.0],
  [3.0, 4.0]
])

grad_output = np.array([1.0, 1.0])
output = min.forward(input)
gradient = min.backward(input, grad_output)
-> print output
[ 1.  3.]
-> print gradient
[[ 1.  0.]
 [ 1.  0.]]
```
---
## Add ##

**Scala:**
```scala
val addLayer = Add(inputSize)
```
**Python:**
```python
add_layer = Add(input_size)
```

A.K.A BiasAdd. This layer adds input tensor with a parameter tensor and output the result.
If the input is 1D, this layer just do a element-wise add. If the input has multiple dimensions,
this layer will treat the first dimension as batch dimension, resize the input tensor to a 2D 
tensor(batch-dimension x input_size) and do a broadcast add between the 2D tensor and the 
parameter.

Please note that the parameter will be trained in the back propagation.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val addLayer = Add(4)
addLayer.bias.set(Tensor(T(1.0f, 2.0f, 3.0f, 4.0f)))
addLayer.forward(Tensor(T(T(1.0f, 1.0f, 1.0f, 1.0f), T(3.0f, 3.0f, 3.0f, 3.0f))))
addLayer.backward(Tensor(T(T(1.0f, 1.0f, 1.0f, 1.0f), T(3.0f, 3.0f, 3.0f, 3.0f))),
    Tensor(T(T(0.1f, 0.1f, 0.1f, 0.1f), T(0.3f, 0.3f, 0.3f, 0.3f))))
```
Gives the output,
```
2.0     3.0     4.0     5.0
4.0     5.0     6.0     7.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]

0.1     0.1     0.1     0.1
0.3     0.3     0.3     0.3
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]
```

**Python example:**
```python
from bigdl.nn.layer import Add
import numpy as np

add_layer = Add(4)
add_layer.set_weights([np.array([1.0, 2.0, 3.0, 4.0])])
add_layer.forward(np.array([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]]))
add_layer.backward(np.array([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]]),
    np.array([[0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3]]))
```
Gives the output,
```
array([[ 2.,  3.,  4.,  5.],
       [ 4.,  5.,  6.,  7.]], dtype=float32)
       
array([[ 0.1       ,  0.1       ,  0.1       ,  0.1       ],
       [ 0.30000001,  0.30000001,  0.30000001,  0.30000001]], dtype=float32)   
```
---
## BiLinear

**Scala:**
```scala
val layer = BiLinear(
  inputSize1,
  inputSize2,
  outputSize,
  biasRes = true,
  wRegularizer = null,
  bRegularizer = null)
```
**Python:**
```python
layer = BiLinear(
    input_size1,
    input_size2,
    output_size,
    bias_res=True,
    wRegularizer=None,
    bRegularizer=None)
```

A bilinear transformation with sparse inputs.
The input tensor given in forward(input) is a table containing both inputs x_1 and x_2,
which are tensors of size N x inputDimension1 and N x inputDimension2, respectively.

Parameters:

* `inputSize1`   dimension of input x_1
* `inputSize2`   dimension of input x_2
* `outputSize`   output dimension
* `biasRes` The layer can be trained without biases by setting bias = false. otherwise true
* `wRegularizer` instance of `Regularizer`
             (eg. L1 or L2 regularization), applied to the input weights matrices.
* `bRegularizer` instance of `Regularizer` applied to the bias.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Bilinear
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val layer = Bilinear(3, 2, 3)
val input1 = Tensor(T(
  T(-1f, 2f, 3f),
  T(-2f, 3f, 4f),
  T(-3f, 4f, 5f)
))
val input2 = Tensor(T(
  T(-2f, 3f),
  T(-1f, 2f),
  T(-3f, 4f)
))
val input = T(input1, input2)

val gradOutput = Tensor(T(
  T(3f, 4f, 5f),
  T(2f, 3f, 4f),
  T(1f, 2f, 3f)
))

val output = layer.forward(input)
val grad = layer.backward(input, gradOutput)

println(output)
-0.14168167	-8.697224	-10.097688
-0.20962894	-7.114827	-8.568602
0.16706467	-19.751905	-24.516418
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

println(grad)
 {
	2: 13.411718	-18.695072
	   14.674414	-19.503393
	   13.9599	-17.271534
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2]
	1: -5.3747015	-17.803686	-17.558662
	   -2.413877	-8.373887	-8.346823
	   -2.239298	-11.249412	-14.537216
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
 }
```

**Python example:**
```python
layer = Bilinear(3, 2, 3)
input_1 = np.array([
  [-1.0, 2.0, 3.0],
  [-2.0, 3.0, 4.0],
  [-3.0, 4.0, 5.0]
])

input_2 = np.array([
  [-3.0, 4.0],
  [-2.0, 3.0],
  [-1.0, 2.0]
])

input = [input_1, input_2]

gradOutput = np.array([
  [3.0, 4.0, 5.0],
  [2.0, 3.0, 4.0],
  [1.0, 2.0, 5.0]
])

output = layer.forward(input)
grad = layer.backward(input, gradOutput)

print output
[[-0.5  1.5  2.5]
 [-1.5  2.5  3.5]
 [-2.5  3.5  4.5]]
[[ 3.  4.  5.]
 [ 2.  3.  4.]
 [ 1.  2.  5.]]

print grad
[array([[ 11.86168194, -14.02727222,  -6.16624403],
       [  6.72984409,  -7.96572971,  -2.89302039],
       [  5.52902842,  -5.76724434,  -1.46646953]], dtype=float32), array([[ 13.22105694,  -4.6879468 ],
       [ 14.39296341,  -6.71434498],
       [ 20.93929482, -13.02455521]], dtype=float32)]
```
---
## Clamp ##

**Scala:**
```scala
val model = Clamp(min, max)
```
**Python:**
```python
model = Clamp(min, max)
```

A kind of hard tanh activition function with integer min and max
* `min` min value
* `max` max value

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

val model = Clamp(-10, 10)
val input = Tensor(2, 2, 2).rand()
val output = model.forward(input)

scala> print(input)
(1,.,.) =
0.95979714	0.27654588	
0.35592428	0.49355772	

(2,.,.) =
0.2624511	0.78833413	
0.967827	0.59160346	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x2]

scala> print(output)
(1,.,.) =
0.95979714	0.27654588	
0.35592428	0.49355772	

(2,.,.) =
0.2624511	0.78833413	
0.967827	0.59160346	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2]

```

**Python example:**
```python
model = Clamp(-10, 10)
input = np.random.randn(2, 2, 2)
output = model.forward(input)

>>> print(input)
[[[-0.66763755  1.15392566]
  [-2.10846048  0.46931736]]

 [[ 1.74174638 -1.04323311]
  [-1.91858729  0.12624046]]]
  
>>> print(output)
[[[-0.66763753  1.15392566]
  [-2.10846043  0.46931735]]

 [[ 1.74174643 -1.04323316]
  [-1.91858733  0.12624046]]
```
---
## Square ##

**Scala:**
```scala
val module = Square()
```
**Python:**
```python
module = Square()
```

Square apply an element-wise square operation.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val module = Square()

println(module.forward(Tensor.range(1, 6, 1)))
```
Gives the output,
```
com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0
4.0
9.0
16.0
25.0
36.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 6]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Square()
print(module.forward(np.arange(1, 7, 1)))
```
Gives the output,
```
[array([  1.,   4.,   9.,  16.,  25.,  36.], dtype=float32)]
```
---
## Mean ##

**Scala:**
```scala
val m = Mean(dimension=1, nInputDims=-1, squeeze=true)
```
**Python:**
```python
m = Mean(dimension=1,n_input_dims=-1, squeeze=True)
```

Mean is a module that simply applies a mean operation over the given dimension - specified by `dimension` (starting from 1).

 
The input is expected to be either one tensor, or a batch of tensors (in mini-batch processing). If the input is a batch of tensors, you need to specify the number of dimensions of each tensor in the batch using `nInputDims`.  When input is one tensor, do not specify `nInputDims` or set it = -1, otherwise input will be interpreted as batch of tensors. 

**Scala example:**
```scala
scala> 
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val input = Tensor(2, 2, 2).randn()
val m1 = Mean()
val output1 = m1.forward(input)
val m2 = Mean(2,1,true)
val output2 = m2.forward(input)

scala> print(input)
(1,.,.) =
-0.52021635     -1.8250599
-0.2321481      -2.5672712

(2,.,.) =
4.007425        -0.8705412
1.6506456       -0.2470611

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x2]

scala> print(output1)
1.7436042       -1.3478005
0.7092488       -1.4071661
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

scala> print(output2)
-0.37618223     -2.1961656
2.8290353       -0.5588012
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input = np.random.rand(2,2,2)
print "input is :",input

m1 = Mean()
out = m1.forward(input)
print "output m1 is :",out

m2 = Mean(2,1,True)
out = m2.forward(input)
print "output m2 is :",out
```
Gives the output,
```python
input is : [[[ 0.01990713  0.37740696]
  [ 0.67689963  0.67715705]]

 [[ 0.45685026  0.58995121]
  [ 0.33405769  0.86351324]]]
creating: createMean
output m1 is : [array([[ 0.23837869,  0.48367909],
       [ 0.50547862,  0.77033514]], dtype=float32)]
creating: createMean
output m2 is : [array([[ 0.34840336,  0.527282  ],
       [ 0.39545399,  0.72673225]], dtype=float32)]
```
---
## Power ##

**Scala:**
```scala
val module = Power(power, scale=1, shift=0)
```
**Python:**
```python
module = Power(power, scale=1.0, shift=0.0)
```

 Apply an element-wise power operation with scale and shift.
 
 `f(x) = (shift + scale * x)^power^`
 
 * `power` the exponent.
 * `scale` Default is 1.
 * `shift` Default is 0.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.Storage
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val power = Power(2, 1, 1)
val input = Tensor(Storage(Array(0.0, 1, 2, 3, 4, 5)), 1, Array(2, 3))
> print(power.forward(input))
1.0	    4.0	     9.0	
16.0	    25.0     36.0	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
from bigdl.nn.layer import *

power = Power(2.0, 1.0, 1.0)
input = np.array([[0.0, 1, 2], [3, 4, 5]])
>power.forward(input)
array([[  1.,   4.,   9.],
       [ 16.,  25.,  36.]], dtype=float32)

```
## CMul ##

**Scala:**
```scala
val module = CMul(size, wRegularizer = null)
```
**Python:**
```python
module = CMul(size, wRegularizer=None)
```

This layer has a weight tensor with given size. The weight will be multiplied element wise to
the input tensor. If the element number of the weight tensor match the input tensor, a simply
element wise multiply will be done. Or the bias will be expanded to the same size of the input.
The expand means repeat on unmatched singleton dimension(if some unmatched dimension isn't
singleton dimension, it will report an error). If the input is a batch, a singleton dimension
will be add to the first dimension before the expand.

  `size` the size of the bias, which is an array of bias shape
  

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val layer = CMul(Array(2, 1))
val input = Tensor(2, 3)
var i = 0
input.apply1(_ => {i += 1; i})
> print(layer.forward(input))
-0.29362988     -0.58725977     -0.88088965
1.9482219       2.4352775       2.9223328
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
from bigdl.nn.layer import *

layer = CMul([2,1])
input = np.array([[1, 2, 3], [4, 5, 6]])
>layer.forward(input)
array([[-0.17618844, -0.35237688, -0.52856529],
       [ 0.85603124,  1.07003903,  1.28404689]], dtype=float32)
```
## AddConstant ##

**Scala:**
```scala
val module = AddConstant(constant_scalar,inplace= false)
```
**Python:**
```python
module = AddConstant(constant_scalar,inplace=False,bigdl_type="float")
```

Element wise add a constant scalar to input tensor
* `constant_scalar` constant value
* `inplace` Can optionally do its operation in-place without using extra state memory
 
**Scala example:**
```scala
val module = AddConstant(3.0)
val input = Tensor(2,3).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.40684703      0.077655114     0.42314094
0.55392265      0.8650696       0.3621729
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

module.forward(input)
res11: com.intel.analytics.bigdl.tensor.Tensor[Float] =
3.406847        3.077655        3.423141
3.5539227       3.8650696       3.3621728
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]

```

**Python example:**
```python
module = AddConstant(3.0,inplace=False,bigdl_type="float")
input = np.array([[1, 2, 3],[4, 5, 6]])
module.forward(input)
[array([
[ 4.,  5.,  6.],
[ 7.,  8.,  9.]], dtype=float32)]
```
---
## Abs ##

**Scala:**
```scala
val m = Abs()
```
**Python:**
```python
m = Abs()
```

An element-wise abs operation.


**Scala example:**
```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val abs = new Abs
val input = Tensor(2)
input(1) = 21f
input(2) = -29f
print(abs.forward(input))
```
`output is:　21.0　29.0`

**Python example:**
```python
abs = Abs()
input = np.array([21, -29, 30])
print(abs.forward(input))
```
`output is: [array([ 21.,  29.,  30.], dtype=float32)]`

---
## Log ##

**Scala:**
```scala
val log = Log()
```
**Python:**
```python
log = Log()
```

The Log module applies a log transformation to the input data

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val log = Log()
val input = Tensor(T(1.0f, Math.E.toFloat))
val gradOutput = Tensor(T(1.0f, 1.0f))
val output = log.forward(input)
val gradient = log.backward(input, gradOutput)
-> print(output)
0.0
1.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2]

-> print(gradient)
1.0
0.36787945
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
import math
log = Log()
input = np.array([1.0, math.e])
grad_output = np.array([1.0, 1.0])
output = log.forward(input)
gradient = log.backward(input, grad_output)

-> print output
[ 0.  1.]

-> print gradient
[ 1.          0.36787945]
```
---
## Sum ##

**Scala:**
```scala
val m = Sum(dimension=1,nInputDims=-1,sizeAverage=false,squeeze=true)
```
**Python:**
```python
m = Sum(dimension=1,n_input_dims=-1,size_average=False,squeeze=True)
```

Sum is a module that simply applies a sum operation over the given dimension - specified by the argument `dimension` (starting from 1). 
 
The input is expected to be either one tensor, or a batch of tensors (in mini-batch processing). If the input is a batch of tensors, you need to specify the number of dimensions of each tensor in the batch using `nInputDims`.  When input is one tensor, do not specify `nInputDims` or set it = -1, otherwise input will be interpreted as batch of tensors. 

**Scala example:**
```scala

scala> 
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val input = Tensor(2, 2, 2).randn()
val m1 = Sum(2)
val output1 = m1.forward(input)
val m2 = Sum(2, 1, true)
val output2 = m2.forward(input)

scala> print(input)
(1,.,.) =
-0.003314678    0.96401167
0.79000163      0.78624517

(2,.,.) =
-0.29975495     0.24742787
0.8709072       0.4381108

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x2]

scala> print(output1)
0.78668696      1.7502568
0.5711522       0.68553865
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

scala> print(output2)
0.39334348      0.8751284
0.2855761       0.34276932
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input=np.random.rand(2,2,2)
print "input is :",input
module = Sum(2)
out = module.forward(input)
print "output 1 is :",out
module = Sum(2,1,True)
out = module.forward(input)
print "output 2 is :",out
```
produces output:
```python
input is : [[[ 0.7194801   0.99120677]
  [ 0.07446639  0.056318  ]]

 [[ 0.08639016  0.17173268]
  [ 0.71686986  0.30503663]]]
creating: createSum
output 1 is : [array([[ 0.7939465 ,  1.04752481],
       [ 0.80325997,  0.47676933]], dtype=float32)]
creating: createSum
output 2 is : [array([[ 0.39697325,  0.5237624 ],
       [ 0.40162998,  0.23838466]], dtype=float32)]
```
---
## Sqrt ##

Apply an element-wise sqrt operation.

**Scala:**

```scala
val sqrt = new Sqrt
```

**Python:**
```python
sqrt = Sqrt()
```

Apply an element-wise sqrt operation.


**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Sqrt
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val input = Tensor(3, 5).range(1, 15, 1)
val sqrt = new Sqrt
val output = sqrt.forward(input)
println(output)

val gradOutput = Tensor(3, 5).range(2, 16, 1)
val gradInput = sqrt.backward(input, gradOutput)
println(gradOutput
```

Gives the output,
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0     1.4142135       1.7320508       2.0     2.236068
2.4494898       2.6457512       2.828427        3.0     3.1622777
3.3166249       3.4641016       3.6055512       3.7416575       3.8729835
```

Gives the gradInput

```
gradInput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0     1.0606601       1.1547005       1.25    1.3416407
1.428869        1.5118579       1.5909902       1.6666667       1.7392527
1.8090681       1.8763883       1.9414507       2.0044594       2.065591
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

sqrt = Sqrt()

input = np.arange(1, 16, 1).astype("float32")
input = input.reshape(3, 5)

output = sqrt.forward(input)
print output

gradOutput = np.arange(2, 17, 1).astype("float32")
gradOutput = gradOutput.reshape(3, 5)

gradInput = sqrt.backward(input, gradOutput)
print gradInput
```

Gives the output,

```
[array([[ 1.        ,  1.41421354,  1.73205078,  2.        ,  2.23606801],
       [ 2.44948983,  2.64575124,  2.82842708,  3.        ,  3.1622777 ],
       [ 3.31662488,  3.46410155,  3.60555124,  3.7416575 ,  3.87298346]], dtype=float32)]
```

Gives the gradInput:

```
[array([[ 1.        ,  1.06066012,  1.15470052,  1.25      ,  1.34164071],
       [ 1.42886901,  1.51185787,  1.59099019,  1.66666675,  1.73925269],
       [ 1.80906808,  1.87638831,  1.94145072,  2.00445938,  2.0655911 ]], dtype=float32)]
```
---
## Exp ##

**Scala:**
```scala
val exp = Exp()
```
**Python:**
```python
exp = Exp()
```

Exp applies element-wise exp operation to input tensor


**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val exp = Exp()
val input = Tensor(3, 3).rand()
> print(input)
0.0858663	0.28117087	0.85724664	
0.62026995	0.29137492	0.07581586	
0.22099794	0.45131826	0.78286386	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]
> print(exp.forward(input))
1.0896606	1.32468		2.356663	
1.85943		1.3382663	1.078764	
1.2473209	1.5703809	2.1877286	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

```

**Python example:**
```python
from bigdl.nn.layer import *
exp = Exp()
> exp.forward(np.array([[1, 2, 3],[1, 2, 3]]))
[array([[  2.71828175,   7.38905621,  20.08553696],
       [  2.71828175,   7.38905621,  20.08553696]], dtype=float32)]

```
---
## Max ##

**Scala:**
```scala
val layer = Max(dim = 1, numInputDims = Int.MinValue)
```
**Python:**
```python
layer = Max(dim, num_input_dims=INTMIN)
```

Applies a max operation over dimension `dim`.

Parameters:

* `dim` max along this dimension

* `numInputDims` Optional. If in a batch model, set to the inputDims.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Max
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val layer = Max(1, 1)
val input = Tensor(T(
  T(-1f, 2f, 3f),
  T(-2f, 3f, 4f),
  T(-3f, 4f, 5f)
))

val gradOutput = Tensor(T(3f, 4f, 5f))

val output = layer.forward(input)
val grad = layer.backward(input, gradOutput)

println(output)
3.0
4.0
5.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]

println(grad)
0.0	0.0	3.0
0.0	0.0	4.0
0.0	0.0	5.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**
```python
layer = Max(1, 1)
input = np.array([
  [-1.0, 2.0, 3.0],
  [-2.0, 3.0, 4.0],
  [-3.0, 4.0, 5.0]
])

gradOutput = np.array([3.0, 4.0, 5.0])

output = layer.forward(input)
grad = layer.backward(input, gradOutput)

print output
[ 3.  4.  5.]

print grad
[[ 0.  0.  3.]
 [ 0.  0.  4.]
 [ 0.  0.  5.]]
```
---
## CAdd ##

**Scala:**
```scala
val module = CAdd(size,bRegularizer=null)
```
**Python:**
```python
module = CAdd(size,bRegularizer=None,bigdl_type="float")
```

This layer has a bias tensor with given size. The bias will be added element wise to the input
tensor. If the element number of the bias tensor match the input tensor, a simply element wise
will be done. Or the bias will be expanded to the same size of the input. The expand means
repeat on unmatched singleton dimension(if some unmatched dimension isn't singleton dimension,
it will report an error). If the input is a batch, a singleton dimension will be add to the first
dimension before the expand.

 * `size` the size of the bias 

**Scala example:**
```scala
val module = CAdd(Array(2, 1),bRegularizer=null)
val input = Tensor(2, 3).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.52146345      0.86262375      0.74210143
0.15882674      0.026310394     0.28394955
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

module.forward(input)
res12: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.97027373      1.311434        1.1909117
-0.047433108    -0.17994945     0.07768971
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
module = CAdd([2, 1],bRegularizer=None,bigdl_type="float")
input = np.random.rand(2, 3)
array([[ 0.71239789,  0.65869477,  0.50425182],
       [ 0.40333312,  0.64843273,  0.07286636]])

module.forward(input)
array([[ 0.89537328,  0.84167016,  0.68722725],
       [ 0.1290929 ,  0.37419251, -0.20137388]], dtype=float32)
```
---
## Cosine ##

**Scala:**
```scala
val m = Cosine(inputSize, outputSize)
```
**Python:**
```python
m = Cosine(input_size, output_size)
```

Cosine is a module used to  calculate the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) of the input to `outputSize` centers, i.e. this layer has the weights `w_j`, for `j = 1,..,outputSize`, where `w_j` are vectors of dimension `inputSize`.

The distance `y_j` between center `j` and input `x` is formulated as `y_j = (x · w_j) / ( || w_j || * || x || )`.

The input given in `forward(input)` must be either a vector (1D tensor) or matrix (2D tensor). If the input is a
vector, it must have the size of `inputSize`. If it is a matrix, then each row is assumed to be an input sample of given batch (the number of rows means the batch size and the number of columns should be equal to the `inputSize`).
	
**Scala example:**
```scala
scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val m = Cosine(2, 3)
val input = Tensor(3, 2).rand()
val output = m.forward(input)

scala> print(input)
0.48958543      0.38529378
0.28814933      0.66979927
0.3581584       0.67365724
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]

scala> print(output)
0.998335        0.9098057       -0.71862763
0.8496431       0.99756527      -0.2993874
0.8901594       0.9999207       -0.37689084
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input=np.random.rand(2,3)
print "input is :",input
module = Cosine(3,3)
module.forward(input)
print "output is :",out
```
Gives the output,

```python
input is : [[ 0.31156943  0.85577626  0.4274042 ]
 [ 0.79744055  0.66431136  0.05657437]]
creating: createCosine
output is : [array([[-0.73284394, -0.28076306, -0.51965958],
       [-0.9563939 , -0.42036989, -0.08060561]], dtype=float32)]


```
---
## Mul ##

**Scala:**
```scala
val module = Mul()
```
**Python:**
```python
module = Mul()
```

Multiply a singla scalar factor to the incoming data

```
                 +----Mul----+
 input -----+---> input * weight -----+----> output
```

**Scala example:**
```scala

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val mul = Mul()

> print(mul.forward(Tensor(1, 5).rand()))
-0.03212923     -0.019040342    -9.136753E-4    -0.014459004    -0.04096878
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

mul = Mul()
input = np.random.uniform(0, 1, (1, 5)).astype("float32")

> mul.forward(input)
[array([[ 0.72429317,  0.7377845 ,  0.09136307,  0.40439236,  0.29011244]], dtype=float32)]

```
---
## MulConstant ##

**Scala:**
```scala
val layer = MulConstant(scalar, inplace)
```
**Python:**
```python
layer = MulConstant(const, inplace)
```

Multiplies input Tensor by a (non-learnable) scalar constant.
This module is sometimes useful for debugging purposes.

Parameters:
* `constant`scalar constant
* `inplace` Can optionally do its operation in-place without using extra state memory. Default: false

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val input = Tensor(T(
 T(1.0f, 2.0f),
 T(3.0f, 4.0f))
)
val gradOutput = Tensor(T(
 T(1.0f, 1.0f),
 T(1.0f, 1.0f))
)
val scalar = 2.0
val module = MulConstant(scalar)
val output = module.forward(input)
val gradient = module.backward(input, gradOutput)
-> print(output)
2.0     4.0     
6.0     8.0     
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
     
-> print(gradient)
2.0     2.0     
2.0     2.0     
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
input = np.array([
          [1.0, 2.0],
          [3.0, 4.0]
        ])
grad_output = np.array([
           [1.0, 1.0],
           [1.0, 1.0]
         ])
scalar = 2.0
module = MulConstant(scalar)
output = module.forward(input)
gradient = module.backward(input, grad_output)
-> print output
[[ 2.  4.]
 [ 6.  8.]]
-> print gradient
[[ 2.  2.]
 [ 2.  2.]]
```

