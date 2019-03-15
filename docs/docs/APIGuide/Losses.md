## L1Cost ##

**Scala:**
```scala
val layer = L1Cost[Float]()
```
**Python:**
```python
layer = L1Cost()
```

Compute L1 norm for input, and sign of input

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.L1Cost
import com.intel.analytics.bigdl.tensor.Tensor

val layer = L1Cost[Float]()
val input = Tensor[Float](2, 2).rand
val target = Tensor[Float](2, 2).rand

val output = layer.forward(input, target)
val gradInput = layer.backward(input, target)

> println(input)
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.48145306      0.476887
0.23729686      0.5169516
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

> println(target)
target: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.42999148      0.22272833
0.49723643      0.17884709
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

> println(output)
output: Float = 1.7125885
> println(gradInput)
gradInput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0     1.0
1.0     1.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
layer = L1Cost()

input = np.random.uniform(0, 1, (2, 2)).astype("float32")
target = np.random.uniform(0, 1, (2, 2)).astype("float32")

output = layer.forward(input, target)
gradInput = layer.backward(input, target)

> output
2.522411
> gradInput
[array([[ 1.,  1.],
        [ 1.,  1.]], dtype=float32)]
```
---
## TimeDistributedCriterion ##

**Scala:**
```scala
val module = TimeDistributedCriterion(critrn, sizeAverage)
```
**Python:**
```python
module = TimeDistributedCriterion(critrn, sizeAverage)
```

This class is intended to support inputs with 3 or more dimensions.
Apply Any Provided Criterion to every temporal slice of an input.
  
* `critrn` embedded criterion
* `sizeAverage` whether to divide the sequence length. Default is false.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.Storage

val criterion = ClassNLLCriterion[Double]()
val layer = TimeDistributedCriterion[Double](criterion, true)
val input = Tensor[Double](Storage(Array(
    1.0262627674932,
    -1.2412600935171,
    -1.0423174168648,
    -1.0262627674932,
    -1.2412600935171,
    -1.0423174168648,
    -0.90330565804228,
    -1.3686840144413,
    -1.0778380454479,
    -0.90330565804228,
    -1.3686840144413,
    -1.0778380454479,
    -0.99131220658219,
    -1.0559142847536,
    -1.2692712660404,
    -0.99131220658219,
    -1.0559142847536,
    -1.2692712660404))).resize(3, 2, 3)
val target = Tensor[Double](3, 2)
    target(Array(1, 1)) = 1
    target(Array(1, 2)) = 1
    target(Array(2, 1)) = 2
    target(Array(2, 2)) = 2
    target(Array(3, 1)) = 3
    target(Array(3, 2)) = 3
> print(layer.forward(input, target))
0.8793184268272332
```

**Python example:**
```python
from bigdl.nn.criterion import *

criterion = ClassNLLCriterion()
layer = TimeDistributedCriterion(criterion, True)
input = np.array([1.0262627674932,
                      -1.2412600935171,
                      -1.0423174168648,
                      -1.0262627674932,
                      -1.2412600935171,
                      -1.0423174168648,
                      -0.90330565804228,
                      -1.3686840144413,
                      -1.0778380454479,
                      -0.90330565804228,
                      -1.3686840144413,
                      -1.0778380454479,
                      -0.99131220658219,
                      -1.0559142847536,
                      -1.2692712660404,
                      -0.99131220658219,
                      -1.0559142847536,
                      -1.2692712660404]).reshape(3,2,3)
target = np.array([[1,1],[2,2],[3,3]])                      
>layer.forward(input, target)
0.8793184
```
---
## MarginRankingCriterion ##

**Scala:**

```scala
val mse = new MarginRankingCriterion(margin=1.0, sizeAverage=true)
```

**Python:**

```python
mse = MarginRankingCriterion(margin=1.0, size_average=true)
```

Creates a criterion that measures the loss given an input `x = {x1, x2}`,
a table of two Tensors of size 1 (they contain only scalars), and a label y (1 or -1).
In batch mode, x is a table of two Tensors of size batchsize, and y is a Tensor of size
batchsize containing 1 or -1 for each corresponding pair of elements in the input Tensor.
If `y == 1` then it assumed the first input should be ranked higher (have a larger value) than
the second input, and vice-versa for `y == -1`.


**Scala example:**

```scala
import com.intel.analytics.bigdl.nn.MarginRankingCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.Storage
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

import scala.util.Random

val input1Arr = Array(1, 2, 3, 4, 5)
val input2Arr = Array(5, 4, 3, 2, 1)

val target1Arr = Array(-1, 1, -1, 1, 1)

val input1 = Tensor(Storage(input1Arr.map(x => x.toFloat)))
val input2 = Tensor(Storage(input2Arr.map(x => x.toFloat)))

val input = T((1.toFloat, input1), (2.toFloat, input2))

val target1 = Tensor(Storage(target1Arr.map(x => x.toFloat)))
val target = T((1.toFloat, target1))

val mse = new MarginRankingCriterion()

val output = mse.forward(input, target)
val gradInput = mse.backward(input, target)

println(output)
println(gradInput)
```
Gives the output

```
output: Float = 0.8                                                                                                                                                                    [21/154]
```

Gives the gradInput,

```
gradInput: com.intel.analytics.bigdl.utils.Table =
 {
        2: -0.0
           0.2
           -0.2
           0.0
           0.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
        1: 0.0
           -0.2
           0.2
           -0.0
           -0.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
 }
```

**Python example:**

```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

mse = MarginRankingCriterion()

input1 = np.array([1, 2, 3, 4, 5]).astype("float32")
input2 = np.array([5, 4, 3, 2, 1]).astype("float32")
input = [input1, input2]

target1 = np.array([-1, 1, -1, 1, 1]).astype("float32")
target = [target1, target1]

output = mse.forward(input, target)
gradInput = mse.backward(input, target)

print output
print gradInput
```

Gives the output,

```
0.8
```

Gives the gradInput,

```
[array([ 0. , -0.2,  0.2, -0. , -0. ], dtype=float32), array([-0. ,  0.2, -0.2,  0. ,  0. ], dtype=float32)] 
```

---
## ClassNLLCriterion

**Scala:**
```scala
val criterion = ClassNLLCriterion(weights = null, sizeAverage = true, logProbAsInput=true)
```
**Python:**
```python
criterion = ClassNLLCriterion(weights=None, size_average=True, logProbAsInput=true)
```

The negative log likelihood criterion. It is useful to train a classification problem with n
classes. If provided, the optional argument weights should be a 1D Tensor assigning weight to
each of the classes. This is particularly useful when you have an unbalanced training set.

The input given through a `forward()` is expected to contain log-probabilities/probabilities of each class:
input has to be a 1D Tensor of size `n`. Obtaining log-probabilities/probabilities in a neural network is easily
achieved by adding a `LogSoftMax`/`SoftMax` layer in the last layer of your neural network. You may use
`CrossEntropyCriterion` instead, if you prefer not to add an extra layer to your network. This
criterion expects a class index (1 to the number of class) as target when calling
`forward(input, target)` and `backward(input, target)`.

 In the log-probabilities case,
 The loss can be described as:
     `loss(x, class) = -x[class]`
 or in the case of the weights argument it is specified as follows:
     `loss(x, class) = -weights[class] * x[class]`
 Due to the behaviour of the backend code, it is necessary to set sizeAverage to false when
 calculating losses in non-batch mode.

 Note that if the target is `-1`, the training process will skip this sample.
 In other words, the forward process will return zero output and the backward process
 will also return zero `gradInput`.

 By default, the losses are averaged over observations for each minibatch. However, if the field
 `sizeAverage` is set to false, the losses are instead summed for each minibatch.

Parameters:

* `weights` weights of each element of the input
* `sizeAverage` A boolean indicating whether normalizing by the number of elements in the input.
                  Default: true
* `logProbAsInput` indicating whether to accept log-probabilities or probabilities as input. True means accepting
               log-probabilities as input.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val criterion = ClassNLLCriterion()
val input = Tensor(T(
              T(1f, 2f, 3f),
              T(2f, 3f, 4f),
              T(3f, 4f, 5f)
          ))

val target = Tensor(T(1f, 2f, 3f))

val loss = criterion.forward(input, target)
val grad = criterion.backward(input, target)

print(loss)
-3.0
println(grad)
-0.33333334	0.0	0.0
0.0	-0.33333334	0.0
0.0	0.0	-0.33333334
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *

criterion = ClassNLLCriterion()
input = np.array([
              [1.0, 2.0, 3.0],
              [2.0, 3.0, 4.0],
              [3.0, 4.0, 5.0]
          ])

target = np.array([1.0, 2.0, 3.0])

loss = criterion.forward(input, target)
gradient= criterion.backward(input, target)

print loss
-3.0
print gradient
-3.0
[[-0.33333334  0.          0.        ]
 [ 0.         -0.33333334  0.        ]
 [ 0.          0.         -0.33333334]]
```

---
## SoftmaxWithCriterion ##

**Scala:**
```scala
val model = SoftmaxWithCriterion(ignoreLabel, normalizeMode)
```
**Python:**
```python
model = SoftmaxWithCriterion(ignoreLabel, normalizeMode)
```

Computes the multinomial logistic loss for a one-of-many classification task, passing real-valued predictions through a softmax to
get a probability distribution over classes. It should be preferred over separate SoftmaxLayer + MultinomialLogisticLossLayer as 
its gradient computation is more numerically stable.

* `ignoreLabel`   (optional) Specify a label value that should be ignored when computing the loss.
* `normalizeMode` How to normalize the output loss.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

val input = Tensor(1, 5, 2, 3).rand()
val target = Tensor(Storage(Array(2.0f, 4.0f, 2.0f, 4.0f, 1.0f, 2.0f))).resize(1, 1, 2, 3)

val model = SoftmaxWithCriterion[Float]()
val output = model.forward(input, target)

> print(input)
(1,1,.,.) =
0.65131104	0.9332143	0.5618989	
0.9965054	0.9370902	0.108070895	

(1,2,.,.) =
0.46066576	0.9636703	0.8123812	
0.31076035	0.16386998	0.37894428	

(1,3,.,.) =
0.49111295	0.3704862	0.9938375	
0.87996656	0.8695406	0.53354675	

(1,4,.,.) =
0.8502225	0.9033509	0.8518651	
0.0692618	0.10121379	0.970959	

(1,5,.,.) =
0.9397213	0.49688303	0.75739735	
0.25074655	0.11416598	0.6594504	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x5x2x3]

> print(output)
1.6689054
```
**Python example:**
```python
input = np.random.randn(1, 5, 2, 3)
target = np.array([[[[2.0, 4.0, 2.0], [4.0, 1.0, 2.0]]]])

model = SoftmaxWithCriterion()
output = model.forward(input, target)

>>> print input
[[[[ 0.78455689  0.01402084  0.82539628]
   [-1.06448238  2.58168413  0.60053703]]

  [[-0.48617618  0.44538094  0.46611658]
   [-1.41509329  0.40038991 -0.63505732]]

  [[ 0.91266769  1.68667933  0.92423611]
   [ 0.1465411   0.84637557  0.14917515]]

  [[-0.7060493  -2.02544114  0.89070726]
   [ 0.14535539  0.73980064 -0.33130613]]

  [[ 0.64538791 -0.44384233 -0.40112523]
   [ 0.44346658 -2.22303621  0.35715986]]]]

>>> print output
2.1002123

```
---
## SmoothL1Criterion ##

**Scala:**
```scala
val slc = SmoothL1Criterion(sizeAverage=true)
```
**Python:**
```python
slc = SmoothL1Criterion(size_average=True)
```
Creates a criterion that can be thought of as a smooth version of the AbsCriterion.
It uses a squared term if the absolute element-wise error falls below 1.
It is less sensitive to outliers than the MSECriterion and in some
cases prevents exploding gradients (e.g. see "Fast R-CNN" paper by Ross Girshick).

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.{Tensor, Storage}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn.SmoothL1Criterion

val slc = SmoothL1Criterion()

val inputArr = Array(
  0.17503996845335,
  0.83220188552514,
  0.48450597329065,
  0.64701424003579,
  0.62694586534053,
  0.34398410236463,
  0.55356747563928,
  0.20383032318205
)
val targetArr = Array(
  0.69956525065936,
  0.86074831243604,
  0.54923197557218,
  0.57388074393384,
  0.63334444304928,
  0.99680578662083,
  0.49997645849362,
  0.23869121982716
)

val input = Tensor(Storage(inputArr.map(x => x.toFloat))).reshape(Array(2, 2, 2))
val target = Tensor(Storage(targetArr.map(x => x.toFloat))).reshape(Array(2, 2, 2))

val output = slc.forward(input, target)
val gradInput = slc.backward(input, target)
```
Gives the output,

```
output: Float = 0.0447365
```

Gives the gradInput,

```
gradInput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.06556566     -0.003568299
-0.008090746    0.009141691

(2,.,.) =
-7.998273E-4    -0.08160271
0.0066988766    -0.0043576136
```


**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

slc = SmoothL1Criterion()

input = np.array([
    0.17503996845335,
    0.83220188552514,
    0.48450597329065,
    0.64701424003579,
    0.62694586534053,
    0.34398410236463,
    0.55356747563928,
    0.20383032318205
])
input.reshape(2, 2, 2)

target = np.array([
    0.69956525065936,
    0.86074831243604,
    0.54923197557218,
    0.57388074393384,
    0.63334444304928,
    0.99680578662083,
    0.49997645849362,
    0.23869121982716
])

target.reshape(2, 2, 2)

output = slc.forward(input, target)
gradInput = slc.backward(input, target)

print output
print gradInput
```
---
## SmoothL1CriterionWithWeights ##

**Scala:**
```scala
val smcod = SmoothL1CriterionWithWeights[Float](sigma: Float = 2.4f, num: Int = 2)
```
**Python:**
```python
smcod = SmoothL1CriterionWithWeights(sigma, num)
```

a smooth version of the AbsCriterion
It uses a squared term if the absolute element-wise error falls below 1.
It is less sensitive to outliers than the MSECriterion and in some cases
prevents exploding gradients (e.g. see "Fast R-CNN" paper by Ross Girshick).

```
   d = (x - y) * w_in
  
  loss(x, y, w_in, w_out)
              | 0.5 * (sigma * d_i)^2 * w_out          if |d_i| < 1 / sigma / sigma
   = 1/n \sum |
              | (|d_i| - 0.5 / sigma / sigma) * w_out   otherwise
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.{Tensor, Storage}
import com.intel.analytics.bigdl.nn.SmoothL1CriterionWithWeights
import com.intel.analytics.bigdl.utils.T

val smcod = SmoothL1CriterionWithWeights[Float](2.4f, 2)

val inputArr = Array(1.1, -0.8, 0.1, 0.4, 1.3, 0.2, 0.2, 0.03)
val targetArr = Array(0.9, 1.5, -0.08, -1.68, -0.68, -1.17, -0.92, 1.58)
val inWArr = Array(-0.1, 1.7, -0.8, -1.9, 1.0, 1.4, 0.8, 0.8)
val outWArr = Array(-1.9, -0.5, 1.9, -1.0, -0.2, 0.1, 0.3, 1.1)

val input = Tensor(Storage(inputArr.map(x => x.toFloat)))
val target = T()
target.insert(Tensor(Storage(targetArr.map(x => x.toFloat))))
target.insert(Tensor(Storage(inWArr.map(x => x.toFloat))))
target.insert(Tensor(Storage(outWArr.map(x => x.toFloat))))
 
val output = smcod.forward(input, target)
val gradInput = smcod.backward(input, target)

> println(output)
  output: Float = -2.17488
> println(gradInput)
-0.010944003
0.425
0.63037443
-0.95
-0.1
0.07
0.120000005
-0.44000003
[com.intel.analytics.bigdl.tensor.DenseTensor of size 8]
```

**Python example:**
```python
smcod = SmoothL1CriterionWithWeights(2.4, 2)

input = np.array([1.1, -0.8, 0.1, 0.4, 1.3, 0.2, 0.2, 0.03]).astype("float32")
targetArr = np.array([0.9, 1.5, -0.08, -1.68, -0.68, -1.17, -0.92, 1.58]).astype("float32")
inWArr = np.array([-0.1, 1.7, -0.8, -1.9, 1.0, 1.4, 0.8, 0.8]).astype("float32")
outWArr = np.array([-1.9, -0.5, 1.9, -1.0, -0.2, 0.1, 0.3, 1.1]).astype("float32")
target = [targetArr, inWArr, outWArr]

output = smcod.forward(input, target)
gradInput = smcod.backward(input, target)

> output
-2.17488
> gradInput
[array([-0.010944  ,  0.42500001,  0.63037443, -0.94999999, -0.1       ,
         0.07      ,  0.12      , -0.44000003], dtype=float32)]
```
---
## MultiMarginCriterion ##

**Scala:**
```scala
val loss = MultiMarginCriterion(p=1,weights=null,margin=1.0,sizeAverage=true)
```
**Python:**
```python
loss = MultiMarginCriterion(p=1,weights=None,margin=1.0,size_average=True)
```

MultiMarginCriterion is a loss function that optimizes a multi-class classification hinge loss (margin-based loss) between input `x` and output `y` (`y` is the target class index).

**Scala example:**
```scala


import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.Storage

val input = Tensor(3,2).randn()
val target = Tensor(Storage(Array(2.0f, 1.0f, 2.0f)))
val loss = MultiMarginCriterion(1)
val output = loss.forward(input,target)
val grad = loss.backward(input,target)

> print(input)
-0.45896783     -0.80141246
0.22560088      -0.13517438
0.2601126       0.35492152
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]

> print(target)
2.0
1.0
2.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]

> print(output)
0.4811434

> print(grad)
0.16666667      -0.16666667
-0.16666667     0.16666667
0.16666667      -0.16666667
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2]


```

**Python example:**
```python
from bigdl.nn.criterion import *
import numpy as np

input  = np.random.randn(3,2)
target = np.array([2,1,2])
print "input=",input
print "target=",target

loss = MultiMarginCriterion(1)
out = loss.forward(input, target)
print "output of loss is : ",out

grad_out = loss.backward(input,target)
print "grad out of loss is : ",grad_out
```
Gives the output,
```
input= [[ 0.46868305 -2.28562261]
 [ 0.8076243  -0.67809689]
 [-0.20342555 -0.66264743]]
target= [2 1 2]
creating: createMultiMarginCriterion
output of loss is :  0.8689213
grad out of loss is :  [[ 0.16666667 -0.16666667]
 [ 0.          0.        ]
 [ 0.16666667 -0.16666667]]


```
---
## HingeEmbeddingCriterion ##


**Scala:**
``` scala
val m = HingeEmbeddingCriterion(margin = 1, sizeAverage = true)
```
**Python:**
```python
m = HingeEmbeddingCriterion(margin=1, size_average=True)
```


Creates a criterion that measures the loss given an input `x` which is a 1-dimensional vector and a label `y` (`1` or `-1`).
This is usually used for measuring whether two inputs are similar or dissimilar, e.g. using the L1 pairwise distance, and is typically used for learning nonlinear embeddings or semi-supervised learning.

```
                 ⎧ x_i,                  if y_i ==  1
loss(x, y) = 1/n ⎨
                 ⎩ max(0, margin - x_i), if y_i == -1
```



**Scala example:**
```scala
import com.intel.analytics.bigdl.utils.{T}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.{T}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val loss = HingeEmbeddingCriterion(1, sizeAverage = false)
val input = Tensor(T(0.1f, 2.0f, 2.0f, 2.0f))
println("input: \n" + input)
println("ouput: ")

println("Target=1: " + loss.forward(input, Tensor(4, 1).fill(1f)))

println("Target=-1: " + loss.forward(input, Tensor(4, 1).fill(-1f)))
```

```
input: 
0.1
2.0
2.0
2.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 4]
ouput: 
Target=1: 6.1
Target=-1: 0.9

```

**Python example:**
```python
import numpy as np
from bigdl.nn.criterion import *
input = np.array([0.1, 2.0, 2.0, 2.0])
target = np.full(4, 1)
print("input: " )
print(input)
print("target: ")
print(target)
print("output: ")
print(HingeEmbeddingCriterion(1.0, size_average= False).forward(input, target))
print(HingeEmbeddingCriterion(1.0, size_average= False).forward(input, np.full(4, -1)))
```
```
input: 
[ 0.1  2.   2.   2. ]
target: 
[1 1 1 1]
output: 
creating: createHingeEmbeddingCriterion
6.1
creating: createHingeEmbeddingCriterion
0.9
```

---
## MarginCriterion ##

**Scala:**
```scala
criterion = MarginCriterion(margin=1.0, sizeAverage=true, squared=false)
```
**Python:**
```python
criterion = MarginCriterion(margin=1.0, sizeAverage=True, squared=False, bigdl_type="float")
```

Creates a criterion that optimizes a two-class classification (squared) hinge loss (margin-based loss) between input x (a Tensor of dimension 1) and output y.
 * `margin` if unspecified, is by default 1.
 * `sizeAverage` whether to average the loss, is by default true
 * `squared` whether to calculate the squared hinge loss

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.MarginCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val criterion = MarginCriterion(margin=1.0, sizeAverage=true)

> val input = Tensor(3, 2).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.33753583      0.3575501
0.23477706      0.7240361
0.92835575      0.4737949
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]

> val target = Tensor(3, 2).rand()
target: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.27280563      0.7022703
0.3348442       0.43332106
0.08935371      0.17876455
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]

> criterion.forward(input, target)
res5: Float = 0.84946966
```

**Python example:**
```python
criterion = MarginCriterion(margin=1.0,size_average=True,bigdl_type="float")
input = np.random.rand(3, 2)
array([[ 0.20824672,  0.67299837],
       [ 0.80561452,  0.19564743],
       [ 0.42501441,  0.19408184]])
       
target = np.random.rand(3, 2)
array([[ 0.67882632,  0.61257846],
       [ 0.10111138,  0.75225082],
       [ 0.60404296,  0.31373273]])
       
criterion.forward(input, target)
0.8166871
```
---
## CosineEmbeddingCriterion ##

**Scala:**
```scala
val cosineEmbeddingCriterion = CosineEmbeddingCriterion(margin  = 0.0, sizeAverage = true)
```
**Python:**
```python
cosineEmbeddingCriterion = CosineEmbeddingCriterion( margin=0.0,size_average=True)
```
CosineEmbeddingCriterion creates a criterion that measures the loss given an input x = {x1, x2},
a table of two Tensors, and a Tensor label y with values 1 or -1.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T
val cosineEmbeddingCriterion = CosineEmbeddingCriterion(0.0, false)
val input1 = Tensor(5).rand()
val input2 = Tensor(5).rand()
val input = T()
input(1.0) = input1
input(2.0) = input2
val target1 = Tensor(Storage(Array(-0.5f)))
val target = T()
target(1.0) = target1

> print(input)
 {
	2.0: 0.4110882
	     0.57726574
	     0.1949834
	     0.67670715
	     0.16984987
	     [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]
	1.0: 0.16878392
	     0.24124223
	     0.8964794
	     0.11156334
	     0.5101486
	     [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]
 }

> print(cosineEmbeddingCriterion.forward(input, target))
0.49919847

> print(cosineEmbeddingCriterion.backward(input, target))
 {
	2: -0.045381278
	   -0.059856333
	   0.72547954
	   -0.2268434
	   0.3842142
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
	1: 0.30369008
	   0.42463788
	   -0.20637506
	   0.5712836
	   -0.06355385
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
 }

```

**Python example:**
```python
from bigdl.nn.layer import *
cosineEmbeddingCriterion = CosineEmbeddingCriterion(0.0, False)
> cosineEmbeddingCriterion.forward([np.array([1.0, 2.0, 3.0, 4.0 ,5.0]),np.array([5.0, 4.0, 3.0, 2.0, 1.0])],[np.array(-0.5)])
0.6363636
> cosineEmbeddingCriterion.backward([np.array([1.0, 2.0, 3.0, 4.0 ,5.0]),np.array([5.0, 4.0, 3.0, 2.0, 1.0])],[np.array(-0.5)])
[array([ 0.07933884,  0.04958678,  0.01983471, -0.00991735, -0.03966942], dtype=float32), array([-0.03966942, -0.00991735,  0.01983471,  0.04958678,  0.07933884], dtype=float32)]

```

---
## BCECriterion ##

**Scala:**
```scala
val criterion = BCECriterion[Float]()
```
**Python:**
```python
criterion = BCECriterion()
```

 This loss function measures the Binary Cross Entropy between the target and the output
``` 
 loss(o, t) = - 1/n sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
```
 or in the case of the weights argument being specified:
```
 loss(o, t) = - 1/n sum_i weights[i] * (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
```
 By default, the losses are averaged for each mini-batch over observations as well as over
 dimensions. However, if the field sizeAverage is set to false, the losses are instead summed.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.BCECriterion
import com.intel.analytics.bigdl.tensor.Tensor

val criterion = BCECriterion[Float]()
val input = Tensor[Float](3, 1).rand

val target = Tensor[Float](3)
target(1) = 1
target(2) = 0
target(3) = 1

val output = criterion.forward(input, target)
val gradInput = criterion.backward(input, target)

> println(target)
res25: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0
0.0
1.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]

> println(output)
output: Float = 0.9009579

> println(gradInput)
gradInput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-1.5277504
1.0736246
-0.336957
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x1]

```

**Python example:**
```python

criterion = BCECriterion()
input = np.random.uniform(0, 1, (3, 1)).astype("float32")
target = np.array([1, 0, 1])
output = criterion.forward(input, target)
gradInput = criterion.backward(input, target)

> output
1.9218739
> gradInput
[array([[-4.3074522 ],
        [ 2.24244714],
        [-1.22368968]], dtype=float32)]

```
---
## DiceCoefficientCriterion ##

**Scala:**
```scala
val loss = DiceCoefficientCriterion(sizeAverage=true, epsilon=1.0f)
```
**Python:**
```python
loss = DiceCoefficientCriterion(size_average=True,epsilon=1.0)
```

DiceCoefficientCriterion is the Dice-Coefficient objective function. 

Both `forward` and `backward` accept two tensors : input and target. The `forward` result is formulated as 
          `1 - (2 * (input intersection target) / (input union target))`

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.Storage

val input = Tensor(2).randn()
val target = Tensor(Storage(Array(2.0f, 1.0f)))
val loss = DiceCoefficientCriterion(epsilon = 1.0f)
val output = loss.forward(input,target)
val grad = loss.backward(input,target)

> print(input)
-0.50278
0.51387966
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2]

> print(target)
2.0
1.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2]

> print(output)
0.9958517

> print(grad)
-0.99619853     -0.49758217
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2]

```

**Python example:**
```python
from bigdl.nn.criterion import *
import numpy as np

input  = np.random.randn(2)
target = np.array([2,1],dtype='float64')

print "input=", input
print "target=", target
loss = DiceCoefficientCriterion(size_average=True,epsilon=1.0)
out = loss.forward(input,target)
print "output of loss is :",out

grad_out = loss.backward(input,target)
print "grad out of loss is :",grad_out
```
produces output:
```python
input= [ 0.4440505  2.9430301]
target= [ 2.  1.]
creating: createDiceCoefficientCriterion
output of loss is : -0.17262316
grad out of loss is : [[-0.38274616 -0.11200322]]
```
---
## MSECriterion ##

**Scala:**
```scala
val criterion = MSECriterion()
```
**Python:**
```python
criterion = MSECriterion()
```

The mean squared error criterion e.g. input: a, target: b, total elements: n
```
loss(a, b) = 1/n * sum(|a_i - b_i|^2)
```

Parameters:

 * `sizeAverage` a boolean indicating whether to divide the sum of squared error by n. 
                 Default: true

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val criterion = MSECriterion()
val input = Tensor(T(
 T(1.0f, 2.0f),
 T(3.0f, 4.0f))
)
val target = Tensor(T(
 T(2.0f, 3.0f),
 T(4.0f, 5.0f))
)
val output = criterion.forward(input, target)
val gradient = criterion.backward(input, target)
-> print(output)
1.0
-> print(gradient)
-0.5    -0.5    
-0.5    -0.5    
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
criterion = MSECriterion()
input = np.array([
          [1.0, 2.0],
          [3.0, 4.0]
        ])
target = np.array([
           [2.0, 3.0],
           [4.0, 5.0]
         ])
output = criterion.forward(input, target)
gradient= criterion.backward(input, target)
-> print output
1.0
-> print gradient
[[-0.5 -0.5]
 [-0.5 -0.5]]
```
---
## SoftMarginCriterion ##

**Scala:**
```scala
val criterion = SoftMarginCriterion(sizeAverage)
```
**Python:**
```python
criterion = SoftMarginCriterion(size_average)
```

Creates a criterion that optimizes a two-class classification logistic loss between
input x (a Tensor of dimension 1) and output y (which is a tensor containing either
1s or -1s).
```
loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x:nElement()
```

Parameters:
* `sizeAverage` A boolean indicating whether normalizing by the number of elements in the input.
                    Default: true

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val criterion = SoftMarginCriterion()
val input = Tensor(T(
 T(1.0f, 2.0f),
 T(3.0f, 4.0f))
)
val target = Tensor(T(
 T(1.0f, -1.0f),
 T(-1.0f, 1.0f))
)
val output = criterion.forward(input, target)
val gradient = criterion.backward(input, target)
-> print(output)
1.3767318
-> print(gradient)
-0.06723536     0.22019927      
0.23814353      -0.0044965525   
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
criterion = SoftMarginCriterion()
input = np.array([
          [1.0, 2.0],
          [3.0, 4.0]
        ])
target = np.array([
           [2.0, 3.0],
           [4.0, 5.0]
         ])
output = criterion.forward(input, target)
gradient = criterion.backward(input, target)
-> print output
1.3767318
-> print gradient
[[-0.06723536  0.22019927]
 [ 0.23814353 -0.00449655]]
```
---
## DistKLDivCriterion ##

**Scala:**
```scala
val loss = DistKLDivCriterion[T](sizeAverage=true)
```
**Python:**
```python
loss = DistKLDivCriterion(size_average=True)
```

DistKLDivCriterion is the Kullback–Leibler divergence loss.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.Storage

val input = Tensor(2).randn()
val target = Tensor(Storage(Array(2.0f, 1.0f)))
val loss = DistKLDivCriterion()
val output = loss.forward(input,target)
val grad = loss.backward(input,target)

> print(input)
-0.3854126
-0.7707398
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2]

> print(target)
2.0
1.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2]

> print(output)
1.4639297

> print(grad)
-1.0
-0.5
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2]

```

**Python example:**
```python
from bigdl.nn.criterion import *
import numpy as np

input  = np.random.randn(2)
target = np.array([2,1])

print "input=", input
print "target=", target
loss = DistKLDivCriterion()
out = loss.forward(input,target)
print "output of loss is :",out

grad_out = loss.backward(input,target)
print "grad out of loss is :",grad_out
```
Gives the output
```python
input= [-1.14333924  0.97662296]
target= [2 1]
creating: createDistKLDivCriterion
output of loss is : 1.348175
grad out of loss is : [-1.  -0.5]
```
---
## ClassSimplexCriterion ##

**Scala:**
```scala
val criterion = ClassSimplexCriterion(nClasses)
```
**Python:**
```python
criterion = ClassSimplexCriterion(nClasses)
```

ClassSimplexCriterion implements a criterion for classification.
It learns an embedding per class, where each class' embedding is a
point on an (N-1)-dimensional simplex, where N is the number of classes.

Parameters:
* `nClasses` An integer, the number of classes.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val criterion = ClassSimplexCriterion(5)
val input = Tensor(T(
 T(1.0f, 2.0f, 3.0f, 4.0f, 5.0f),
 T(4.0f, 5.0f, 6.0f, 7.0f, 8.0f)
))
val target = Tensor(2)
target(1) = 2.0f
target(2) = 1.0f
val output = criterion.forward(input, target)
val gradient = criterion.backward(input, target)
-> print(output)
23.562702
-> print(gradient)
0.25    0.20635083      0.6     0.8     1.0     
0.6     1.0     1.2     1.4     1.6     
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x5]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
criterion = ClassSimplexCriterion(5)
input = np.array([
   [1.0, 2.0, 3.0, 4.0, 5.0],
   [4.0, 5.0, 6.0, 7.0, 8.0]
])
target = np.array([2.0, 1.0])
output = criterion.forward(input, target)
gradient = criterion.backward(input, target)
-> print output
23.562702
-> print gradient
[[ 0.25        0.20635083  0.60000002  0.80000001  1.        ]
 [ 0.60000002  1.          1.20000005  1.39999998  1.60000002]]
```
---
## L1HingeEmbeddingCriterion ##

**Scala:**
```scala
val model = L1HingeEmbeddingCriterion(margin)
```
**Python:**
```python
model = L1HingeEmbeddingCriterion(margin)
```

Creates a criterion that measures the loss given an input ``` x = {x1, x2} ```, a table of two Tensors, and a label y (1 or -1).
This is used for measuring whether two inputs are similar or dissimilar, using the L1 distance, and is typically used for learning nonlinear embeddings or semi-supervised learning.
```
             ⎧ ||x1 - x2||_1,                  if y ==  1
loss(x, y) = ⎨
             ⎩ max(0, margin - ||x1 - x2||_1), if y == -1
```
The margin has a default value of 1, or can be set in the constructor.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val model = L1HingeEmbeddingCriterion(0.6)
val input1 = Tensor(T(1.0f, -0.1f))
val input2 = Tensor(T(2.0f, -0.2f))
val input = T(input1, input2)
val target = Tensor(1)
target(Array(1)) = 1.0f

val output = model.forward(input, target)

scala> print(output)
1.1
```

**Python example:**
```python
model = L1HingeEmbeddingCriterion(0.6)
input1 = np.array(1.0, -0.1)
input2 = np.array(2.0, -0.2)
input = [input1, input2]
target = np.array([1.0])

output = model.forward(input, target)

>>> print output
1.1
```
---
## CrossEntropyCriterion ##

**Scala:**
```scala
val module = CrossEntropyCriterion(weights, sizeAverage)
```
**Python:**
```python
module = CrossEntropyCriterion(weights, sizeAverage)
```

This criterion combines LogSoftMax and ClassNLLCriterion in one single class.

* `weights` A tensor assigning weight to each of the classes
* `sizeAverage` whether to divide the sequence length. Default is true.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.Storage

val layer = CrossEntropyCriterion[Double]()
val input = Tensor[Double](Storage(Array(
    1.0262627674932,
    -1.2412600935171,
    -1.0423174168648,
    -0.90330565804228,
    -1.3686840144413,
    -1.0778380454479,
    -0.99131220658219,
    -1.0559142847536,
    -1.2692712660404
    ))).resize(3, 3)
val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3
> print(layer.forward(input, target))
0.9483051199107635
```

**Python example:**
```python
from bigdl.nn.criterion import *

layer = CrossEntropyCriterion()
input = np.array([1.0262627674932,
                      -1.2412600935171,
                      -1.0423174168648,
                      -0.90330565804228,
                      -1.3686840144413,
                      -1.0778380454479,
                      -0.99131220658219,
                      -1.0559142847536,
                      -1.2692712660404
                      ]).reshape(3,3)
target = np.array([1, 2, 3])                      
>layer.forward(input, target)
0.94830513
```
---
## ParallelCriterion ##

**Scala:**

```scala
val pc = ParallelCriterion(repeatTarget=false)
```

**Python:**

```python
pc = ParallelCriterion(repeat_target=False)
```

ParallelCriterion is a weighted sum of other criterions each applied to a different input
and target. Set repeatTarget = true to share the target for criterions.
Use add(criterion[, weight]) method to add criterion. Where weight is a scalar(default 1).

**Scala example:**

```scala
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.{Tensor, Storage}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn.{ParallelCriterion, ClassNLLCriterion, MSECriterion}

val pc = ParallelCriterion()

val input = T(Tensor(2, 10), Tensor(2, 10))
var i = 0
input[Tensor[Float]](1).apply1(_ => {i += 1; i})
input[Tensor[Float]](2).apply1(_ => {i -= 1; i})
val target = T(Tensor(Storage(Array(1.0f, 8.0f))), Tensor(2, 10).fill(1.0f))

val nll = ClassNLLCriterion()
val mse = MSECriterion()
pc.add(nll, 0.5).add(mse)

val output = pc.forward(input, target)
val gradInput = pc.backward(input, target)

println(output)
println(gradInput)

```
Gives the output,

```
100.75

```

Gives the gradInput,

```
 {
        2: 1.8000001    1.7     1.6     1.5     1.4     1.3000001       1.2     1.1     1.0     0.90000004
           0.8  0.7     0.6     0.5     0.4     0.3     0.2     0.1     0.0     -0.1
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x10]
        1: -0.25        0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0
           0.0  0.0     0.0     0.0     0.0     0.0     0.0     -0.25   0.0     0.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x10]
 }

```
**Python example:**

```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

pc = ParallelCriterion()

input1 = np.arange(1, 21, 1).astype("float32")
input2 = np.arange(0, 20, 1).astype("float32")[::-1]
input1 = input1.reshape(2, 10)
input2 = input2.reshape(2, 10)

input = [input1, input2]

target1 = np.array([1.0, 8.0]).astype("float32")
target1 = target1.reshape(2)
target2 = np.full([2, 10], 1).astype("float32")
target2 = target2.reshape(2, 10)
target = [target1, target2]

nll = ClassNLLCriterion()
mse = MSECriterion()

pc.add(nll, weight = 0.5).add(mse)

print "input = \n %s " % input
print "target = \n %s" % target

output = pc.forward(input, target)
gradInput = pc.backward(input, target)

print "output = %s " % output
print "gradInput = %s " % gradInput
```
Gives the output,

```
input = 
 [array([[  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.],
       [ 11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.]], dtype=float32), array([[ 19.,  18.,  17.,  16.,  15.,  14.,  13.,  12.,  11.,  10.],
       [  9.,   8.,   7.,   6.,   5.,   4.,   3.,   2.,   1.,   0.]], dtype=float32)] 
target = 
 [array([ 1.,  8.], dtype=float32), array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]], dtype=float32)]
output = 100.75 
gradInput = [array([[-0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.25,  0.  ,  0.  ]], dtype=float32), array([[ 1.80000007,  1.70000005,  1.60000002,  1.5       ,  1.39999998,
         1.30000007,  1.20000005,  1.10000002,  1.        ,  0.90000004],
       [ 0.80000001,  0.69999999,  0.60000002,  0.5       ,  0.40000001,
         0.30000001,  0.2       ,  0.1       ,  0.        , -0.1       ]], dtype=float32)]
```
---
## MultiLabelMarginCriterion ##

**Scala:**
```scala
val multiLabelMarginCriterion = MultiLabelMarginCriterion(sizeAverage = true)
```
**Python:**
```python
multiLabelMarginCriterion = MultiLabelMarginCriterion(size_average=True)
```
MultiLabelMarginCriterion creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss) between input x and output y 

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val multiLabelMarginCriterion = MultiLabelMarginCriterion(false)
val input = Tensor(4).rand()
val target = Tensor(4)
target(Array(1)) = 3
target(Array(2)) = 2
target(Array(3)) = 1
target(Array(4)) = 0

> print(input)
0.40267515
0.5913795
0.84936756
0.05999674

>  print(multiLabelMarginCriterion.forward(input, target))
0.33414197

> print(multiLabelMarginCriterion.backward(input, target))
-0.25
-0.25
-0.25
0.75
[com.intel.analytics.bigdl.tensor.DenseTensor of size 4]


```

**Python example:**
```python
from bigdl.nn.layer import *
multiLabelMarginCriterion = MultiLabelMarginCriterion(False)

> multiLabelMarginCriterion.forward(np.array([0.3, 0.4, 0.2, 0.6]), np.array([3, 2, 1, 0]))
0.975

> multiLabelMarginCriterion.backward(np.array([0.3, 0.4, 0.2, 0.6]), np.array([3, 2, 1, 0]))
[array([-0.25, -0.25, -0.25,  0.75], dtype=float32)]

```

---
## MultiLabelSoftMarginCriterion ##

**Scala:**
```scala
val criterion = MultiLabelSoftMarginCriterion(weights = null, sizeAverage = true)
```
**Python:**
```python
criterion = MultiLabelSoftMarginCriterion(weights=None, size_average=True)
```

MultiLabelSoftMarginCriterion is a multiLabel multiclass criterion based on sigmoid:
```
l(x,y) = - sum_i y[i] * log(p[i]) + (1 - y[i]) * log (1 - p[i])
```
 where ```p[i] = exp(x[i]) / (1 + exp(x[i]))```
 
 If with weights,
 ```
 l(x,y) = - sum_i weights[i] (y[i] * log(p[i]) + (1 - y[i]) * log (1 - p[i]))
 ```

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val criterion = MultiLabelSoftMarginCriterion()
val input = Tensor(3)
input(Array(1)) = 0.4f
input(Array(2)) = 0.5f
input(Array(3)) = 0.6f
val target = Tensor(3)
target(Array(1)) = 0
target(Array(2)) = 1
target(Array(3)) = 1

> criterion.forward(input, target)
res0: Float = 0.6081934
```

**Python example:**
```python
from bigdl.nn.criterion import *
import numpy as np

criterion = MultiLabelSoftMarginCriterion()
input = np.array([0.4, 0.5, 0.6])
target = np.array([0, 1, 1])

> criterion.forward(input, target)
0.6081934
```
---
## AbsCriterion ##

**Scala:**
```scala
val criterion = AbsCriterion(sizeAverage)
```
**Python:**
```python
criterion = AbsCriterion(sizeAverage)
```

Measures the mean absolute value of the element-wise difference between input and target

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val criterion = AbsCriterion()
val input = Tensor(T(1.0f, 2.0f, 3.0f))
val target = Tensor(T(4.0f, 5.0f, 6.0f))
val output = criterion.forward(input, target)

> print(output)
3.0
```

**Python example:**
```python
criterion = AbsCriterion()
input = np.array([1.0, 2.0, 3.0])
target = np.array([4.0, 5.0, 6.0])
output=criterion.forward(input, target)

>>> print output
3.0
```
---
## MultiCriterion ##

**Scala:**
```scala
val criterion = MultiCriterion()
```
**Python:**
```python
criterion = MultiCriterion()
```

MultiCriterion is a weighted sum of other criterions each applied to the same input and target

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val criterion = MultiCriterion()
val nll = ClassNLLCriterion()
val mse = MSECriterion()
criterion.add(nll, 0.5)
criterion.add(mse)

val input = Tensor(5).randn()
val target = Tensor(5)
target(Array(1)) = 1
target(Array(2)) = 2
target(Array(3)) = 3
target(Array(4)) = 2
target(Array(5)) = 1

val output = criterion.forward(input, target)

> input
1.0641425
-0.33507252
1.2345984
0.08065767
0.531199
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]


> output
res7: Float = 1.9633228
```

**Python example:**
```python
from bigdl.nn.criterion import *
import numpy as np

criterion = MultiCriterion()
nll = ClassNLLCriterion()
mse = MSECriterion()
criterion.add(nll, 0.5)
criterion.add(mse)

input = np.array([0.9682213801388531,
0.35258855644097503,
0.04584479998452568,
-0.21781499692588918,
-1.02721844006879])
target = np.array([1, 2, 3, 2, 1])

output = criterion.forward(input, target)

> output
3.6099546
```

## GaussianCriterion ##

**Scala:**
```scala
val criterion = GaussianCriterion()
```
**Python:**
```python
criterion = GaussianCriterion()
```

GaussianCriterion computes the log-likelihood of a sample given a Gaussian distribution.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn.GaussianCriterion
import com.intel.analytics.bigdl.utils.T

val criterion = GaussianCriterion()

val input1 = Tensor[Float](2, 3).range(1, 6, 1)
val input2 = Tensor[Float](2, 3).range(1, 12, 2)
val input = T(input1, input2)

val target = Tensor[Float](2, 3).range(2, 13, 2)

val loss = criterion.forward(input, target)

> loss
loss: Float = 23.836603
```

**Python example:**
```python
import numpy as np
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

criterion = GaussianCriterion()

input1 = np.arange(1, 7, 1).astype("float32")
input2 = np.arange(1, 12, 2).astype("float32")
input1 = input1.reshape(2, 3)
input2 = input2.reshape(2, 3)
input = [input1, input2]

target = np.arange(2, 13, 2).astype("float32")
target = target.reshape(2, 3)

loss = criterion.forward(input, target)

> output
23.836603
```

## KLDCriterion ##

**Scala:**
```scala
val criterion = KLDCriterion()
```
**Python:**
```python
criterion = KLDCriterion()
```

Computes the KL-divergence of the input normal distribution to a standard normal distribution.
The input has to be a table. The first element of input is the mean of the distribution,
the second element of input is the log_variance of the distribution. The input distribution is
assumed to be diagonal.

The mean and log_variance are both assumed to be two dimensional tensors. The first dimension are
interpreted as batch. The output is the average/sum of each observation

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn.KLDCriterion
import com.intel.analytics.bigdl.utils.T

val criterion = KLDCriterion()

val input1 = Tensor[Float](2, 3).range(1, 6, 1)
val input2 = Tensor[Float](2, 3).range(1, 12, 2)
val input = T(input1, input2)

val target = Tensor[Float](2, 3).range(2, 13, 2)

val loss = criterion.forward(input, target)

> loss
loss: Float = 34647.04
```

**Python example:**
```python
import numpy as np
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

criterion = KLDCriterion()

input1 = np.arange(1, 7, 1).astype("float32")
input2 = np.arange(1, 12, 2).astype("float32")
input1 = input1.reshape(2, 3)
input2 = input2.reshape(2, 3)
input = [input1, input2]

target = np.arange(2, 13, 2).astype("float32")
target = target.reshape(2, 3)

loss = criterion.forward(input, target)

> loss
34647.04
```

## CosineProximityCriterion ##

**Scala:**
```scala
val criterion = CosineProximityCriterion()
```
**Python:**
```python
criterion = CosineProximityCriterion()
```

Computes the negative of the mean cosine proximity between predictions and targets.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn.CosineProximityCriterion

val criterion = CosineProximityCriterion()

val input = Tensor[Float](2, 3).rand()

val target = Tensor[Float](2, 3).rand()

val loss = criterion.forward(input, target)

> loss
loss: Float = -0.28007346
```

**Python example:**
```python
import numpy as np
from bigdl.nn.criterion import *

criterion = CosineProximityCriterion()

input = np.arange(1, 7, 1).astype("float32")
input = input.reshape(2, 3)
target = np.arange(2, 13, 2).astype("float32")
target = target.reshape(2, 3)

loss = criterion.forward(input, target)

> loss
-0.3333333
```

## MeanSquaredLogarithmicCriterion ##
**Scala:**
```scala
val criterion = MeanSquaredLogarithmicCriterion()
```
**Python:**
```python
criterion = MeanSquaredLogarithmicCriterion()
```

compute mean squared logarithmic error for input and target

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn.MeanSquaredLogarithmicCriterion
import com.intel.analytics.bigdl.utils.T

val criterion = MeanSquaredLogarithmicCriterion()
val input = Tensor[Float](2, 3).range(1, 6, 1)
val target = Tensor[Float](2, 3).range(2, 13, 2)
val loss = criterion.forward(input, target)

> loss
loss: Float = 0.30576965
```

**Python example:**
```python
import numpy as np
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

criterion = MeanSquaredLogarithmicCriterion()

input = np.arange(1, 7, 1).astype("float32")
input = input.reshape(2, 3)
target = np.arange(2, 13, 2).astype("float32")
target = target.reshape(2, 3)

loss = criterion.forward(input, target)

> loss
0.30576965
```

## MeanAbsolutePercentageCriterion ##
**Scala:**
```scala
val criterion = MeanAbsolutePercentageCriterion()
```
**Python:**
```python
criterion = MeanAbsolutePercentageCriterion()
```

compute mean absolute percentage error for intput and target

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn.MeanAbsolutePercentageCriterion
import com.intel.analytics.bigdl.utils.T

val criterion = MeanAbsolutePercentageCriterion()

val input = Tensor[Float](2, 3).range(1, 6, 1)
val target = Tensor[Float](2, 3).range(2, 13, 2)
val loss = criterion.forward(input, target)

> loss
loss: Float = 50.0
```

**Python example:**
```python
import numpy as np
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

criterion = MeanAbsolutePercentageCriterion()

input = np.arange(1, 7, 1).astype("float32")
input = input.reshape(2, 3)
target = np.arange(2, 13, 2).astype("float32")
target = target.reshape(2, 3)

loss = criterion.forward(input, target)

> loss
50.0
```


## KullbackLeiblerDivergenceCriterion ##
**Scala:**
```scala
val criterion = KullbackLeiblerDivergenceCriterion()
```
**Python:**
```python
criterion = KullbackLeiblerDivergenceCriterion()
```

compute Kullback Leibler Divergence Criterion error for intput and target

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn.KullbackLeiblerDivergenceCriterion
import com.intel.analytics.bigdl.utils.T

val criterion = KullbackLeiblerDivergenceCriterion[Float]()
val input = Tensor[Float](Array(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f), Array(2, 3))
val target = Tensor[Float](Array(0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f), Array(2, 3))
val loss = criterion.forward(input, target)

> loss
loss: Float = 0.59976757
```

**Python example:**
```python
import numpy as np
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

criterion = KullbackLeiblerDivergenceCriterion()

y_pred = np.matrix('0.1 0.2 0.3; 0.4 0.5 0.6')
y_true = np.matrix('0.6 0.5 0.4; 0.3 0.2 0.1')

loss = criterion.forward(y_pred, y_true)

> loss
0.59976757
```

## PoissonCriterion ##
**Scala:**
```scala
val criterion = PoissonCriterion()
```
**Python:**
```python
criterion = PoissonCriterion()
```

compute Poisson error for intput and target

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn.PoissonCriterion
import com.intel.analytics.bigdl.utils.T

val criterion = PoissonCriterion()
val input = Tensor[Float](2, 3).range(1, 6, 1)
val target = Tensor[Float](2, 3).range(2, 13, 2)
val loss = criterion.forward(input, target)

> loss
loss = -6.1750183

```

**Python example:**
```python
import numpy as np
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

criterion = PoissonCriterion()
input = np.arange(1, 7, 1).astype("float32")
input = input.reshape(2, 3)
target = np.arange(2, 13, 2).astype("float32")
target = target.reshape(2, 3)

loss = criterion.forward(input, target)

> loss
-6.1750183
```


## TransformerCriterion ##
**Scala:**
```scala
val criterion = TransformerCriterion(criterion, Some(inputTransformer), Some(targetTransformer))
```
**Python:**
```python
criterion = TransformerCriterion(criterion, input_transformer, targetTransformer)
```

The criterion that takes two modules (optional) to transform input and target, and take
one criterion to compute the loss with the transformed input and target.

This criterion can be used to construct complex criterion. For example, the
`inputTransformer` and `targetTransformer` can be pre-trained CNN networks,
and we can use the networks' output to compute the high-level feature
reconstruction loss, which is commonly used in areas like neural style transfer
(https://arxiv.org/abs/1508.06576), texture synthesis (https://arxiv.org/abs/1505.07376),
.etc.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn.TransformerCriterion
import com.intel.analytics.bigdl.utils.T

val criterion = MSECriterion()
val input = Tensor[Float](2, 3).range(1, 6, 1)
val target = Tensor[Float](2, 3).range(2, 13, 2)
val inputTransformer = Identity()
val targetTransformer = Identity()
val transCriterion = TransformerCriterion(criterion,
     Some(inputTransformer), Some(targetTransformer))
val loss = transCriterion.forward(input, target)

> loss
15.166667

```

**Python example:**
```python
import numpy as np
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

criterion = MSECriterion()
input = np.arange(1, 7, 1).astype("float32")
input = input.reshape(2, 3)
target = np.arange(2, 13, 2).astype("float32")
target = target.reshape(2, 3)

inputTransformer = Identity()
targetTransformer = Identity()
transCriterion = TransformerCriterion(criterion, inputTransformer, targetTransformer)
loss = transCriterion.forward(input, target)


> loss
15.166667
```

## DotProductCriterion ##
**Scala:**
```scala
val criterion = DotProductCriterion(sizeAverage=false)
```
**Python:**
```python
criterion = DotProductCriterion(sizeAverage=False)
```

Compute the dot product of input and target tensor.
Input and target are required to have the same size.

* sizeAverage:  whether to average over each observations in the same batch

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val criterion = DotProductCriterion()
val input = Tensor[Float](2, 3).range(1, 6, 1)
val target = Tensor[Float](2, 3).range(2, 13, 2)

val loss = criterion.forward(input, target)

> loss
182.0

```

**Python example:**
```python
import numpy as np
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

criterion = DotProductCriterion()
input = np.arange(1, 7, 1).astype("float32")
input = input.reshape(2, 3)
target = np.arange(2, 13, 2).astype("float32")
target = target.reshape(2, 3)

loss = criterion.forward(input, target)


> loss
182.0
```

## PGCriterion ##
**Scala:**
```scala
val criterion = PGCriterion(sizeAverage=false)
```
**Python:**
```python
criterion = PGCriterion(sizeAverage=False)
```

The Criterion to compute the negative policy gradient given a
multinomial distribution and the sampled action and reward.

The input to this criterion should be a 2-D tensor representing
a batch of multinomial distribution, the target should also be
a 2-D tensor with the same size of input, representing the sampled
action and reward/advantage with the index of non-zero element in the vector
represents the sampled action and the non-zero element itself represents
the reward. If the action is space is large, you should consider using
SparseTensor for target.

The loss computed is simple the standard policy gradient,

   loss = - 1/n * sum(R_{n} dot_product log(P_{n}))

 where R_{n} is the reward vector, and P_{n} is the input distribution.
 
 
* sizeAverage:  whether to average over each observations in the same batch

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val criterion = PGCriterion()
val input = Tensor[Float](2, 3).range(1, 6, 1)
val target = Tensor[Float](2, 3).range(2, 13, 2)

val loss = criterion.forward(input, target)

> loss
-58.05011

```

**Python example:**
```python
import numpy as np
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

criterion = PGCriterion()
input = np.arange(1, 7, 1).astype("float32")
input = input.reshape(2, 3)
target = np.arange(2, 13, 2).astype("float32")
target = target.reshape(2, 3)

loss = criterion.forward(input, target)


> loss
-58.05011
```

## CosineDistanceCriterion ##

**Scala:**
```scala
val criterion = CosineDistanceCriterion()
```
**Python:**
```python
criterion = CosineDistanceCriterion(size_average = True)
```

 This loss function measures the Cosine Distance between the target and the output
``` 
 loss(o, t) = 1 - cos(o, t)
```
 By default, the losses are averaged for each mini-batch over observations as well as over
 dimensions. However, if the field sizeAverage is set to false, the losses are instead summed.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat

val criterion = CosineDistanceCriterion()
val input = Tensor(1, 5).rand
val target = Tensor(1, 5).rand
val loss = criterion.forward(input, target)

> println(target)
target: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.95363826	0.3175587	0.90366143	0.10316128	0.05317958
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5]


> println(input)
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.5895327	0.20547494	0.43118918	0.28824696	0.032088008
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5]


> println(loss)
loss: Float = 0.048458755

```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

criterion = CosineDistanceCriterion(size_average = True)
input = np.random.uniform(0, 1, (1, 5)).astype("float32")
target = np.random.uniform(0, 1, (1, 5)).astype("float32")
loss = criterion.forward(input, target)

> input
array([[ 0.34291017,  0.95894575,  0.23869193,  0.42518589,  0.73902631]], dtype=float32)

> target 
array([[ 0.00489056,  0.7253111 ,  0.94344038,  0.69811821,  0.45532107]], dtype=float32)

> loss
0.20651573

```

## CategoricalCrossEntropy ##

**Scala:**
```scala
val criterion = CategoricalCrossEntropy()
```
**Python:**
```python
criterion = CategoricalCrossEntropy()
```

This is same with cross entropy criterion, except the target tensor is a
one-hot tensor.

**Scala example:**
```scala
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

val criterion = CategoricalCrossEntropy()
val input = Tensor(1, 5).rand()
val target = Tensor(1, 5).zero().setValue(1, 3, 1)
val loss = criterion.forward(input, target)

> println(target)
target: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.0     0.0     1.0     0.0     0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5]


> println(input)
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.95245546      0.8304343       0.8296352       0.13989972      0.17466335
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5]


> println(loss)
loss: Float = 1.2607772
```

**Python example:**
```python
import numpy as np
from bigdl.nn.criterion import *

criterion = CategoricalCrossEntropy()
input = np.random.uniform(0, 1, (1, 5)).astype("float32")
target = np.zeros((1, 5)).astype("float32")
target[0, 2] = 1
loss = criterion.forward(input, target)

> input
array([[ 0.31309742,  0.75959802,  0.01649681,  0.65792692,  0.21528937]], dtype=float32)

> target 
array([[ 0.,  0.,  1.,  0.,  0.]], dtype=float32)

> loss
4.7787604
```
