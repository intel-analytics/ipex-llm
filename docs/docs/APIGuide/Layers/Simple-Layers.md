## Linear ##

**Scala:**
```scala
val module = Linear(
  inputSize,
  outputSize,
  withBias = true,
  wRegularizer = null,
  bRegularizer = null,
  initWeight = null,
  initBias = null,
  initGradWeight = null,
  initGradBias = null)
```
**Python:**
```python
module = Linear(
  input_size,
  output_size,
  init_method="default",
  with_bias=True,
  wRegularizer=None,
  bRegularizer=None,
  init_weight=None,
  init_bias=None,
  init_grad_weight=None,
  init_grad_bias=None)
```

The `Linear` module applies a linear transformation to the input data,
i.e. `y = Wx + b`. The `input` given in `forward(input)` must be either
a vector (1D tensor) or matrix (2D tensor). If the input is a vector, it must
have the size of `inputSize`. If it is a matrix, then each row is assumed to be
an input sample of given batch (the number of rows means the batch size and
the number of columns should be equal to the `inputSize`).

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val module = Linear(3, 5)

println(module.forward(Tensor.range(1, 3, 1)))
```

Gives the output,
```
com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.79338956
-2.3417668
-2.7557678
-0.07507719
-1.009765
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Linear(3, 5)

print(module.forward(np.arange(1, 4, 1)))
```
Gives the output,
```
[array([ 0.31657887, -1.11062765, -1.16235781, -0.67723978,  0.74650359], dtype=float32)]
```
---
## Reverse ##


**Scala:**
```scala
val m = Reverse(dim = 1, isInplace = false)
```
**Python:**
```python
m = Reverse(dimension=1)
```

 Reverse the input w.r.t given dimension.
 The input can be a Tensor or Table. `Dimension` is one-based index.
 

**Scala example:**
```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

def randomn(): Double = RandomGenerator.RNG.uniform(0, 1)
val input = Tensor(2, 3)
input.apply1(x => randomn().toFloat)
println("input:")
println(input)
val layer = new Reverse(1)
println("output:")
println(layer.forward(input))
```

```
input:
0.17271264898590744	0.019822501810267568	0.18107921979390085	
0.4003877849318087	0.5567442716564983	0.14120339532382786	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
output:
0.4003877849318087	0.5567442716564983	0.14120339532382786	
0.17271264898590744	0.019822501810267568	0.18107921979390085	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]

```

**Python example:**
```python
input = np.random.random((2,3))
layer = Reverse(1)
print("input:")
print(input)
print("output:")
print(layer.forward(input))
```
```
creating: createReverse
input:
[[ 0.89089717  0.07629756  0.30863782]
 [ 0.16066851  0.06421963  0.96719367]]
output:
[[ 0.16066851  0.06421963  0.96719366]
 [ 0.89089715  0.07629756  0.30863783]]

 
```
---
## Reshape ##

**Scala:**
```scala
val reshape = Reshape(size, batchMode)
```
**Python:**
```python
reshape = Reshape(size, batch_mode)
```

The `forward(input)` reshape the input tensor into `size(0) * size(1) * ...` tensor,
taking the elements row-wise.

parameters:
* `size` the size after reshape
* `batchMode` It is an optional argument. If it is set to `Some(true)`,
                  the first dimension of input is considered as batch dimension,
                  and thus keep this dimension size fixed. This is necessary
                  when dealing with batch sizes of one. When set to `Some(false)`,
                  it forces the entire input (including the first dimension) to be reshaped
                  to the input size. Default is `None`, which means the module considers
                  inputs with more elements than the product of provided sizes (`size(0) *
                  size(1) * ..`) to be batches, otherwise in no batch mode.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val reshape = Reshape(Array(3, 2))
val input = Tensor(2, 2, 3).rand()
val output = reshape.forward(input)
-> print(output.size().toList)      
List(2, 3, 2)
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
reshape =  Reshape([3, 2])
input = np.random.rand(2, 2, 3)
output = reshape.forward(input)
-> print output[0].shape
(2, 3, 2)
```
---
## Index ##

**Scala:**
```scala
val model = Index(dimension)
```
**Python:**
```python
model = Index(dimension)
```
Applies the Tensor index operation along the given dimension.

**Scala example:**

```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

val input1 = Tensor(3).rand()
val input2 = Tensor(4)
input2(Array(1)) = 1.0f
input2(Array(2)) = 2.0f
input2(Array(3)) = 2.0f
input2(Array(4)) = 3.0f

val input = T(input1, input2)
val model = Index(1)
val output = model.forward(input)

scala> print(input)
 {
	2: 1.0
	   2.0
	   2.0
	   3.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 4]
	1: 0.124325536
	   0.8768922
	   0.6378146
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
 }
scala> print(output)
0.124325536
0.8768922
0.8768922
0.6378146
[com.intel.analytics.bigdl.tensor.DenseTensor of size 4]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np
 
input1 = np.random.randn(3)
input2 = np.array([1, 2, 2, 3])
input = [input1, input2]

model = Index(1)
output = model.forward(input)

>>> print(input)
[array([-0.45804847, -0.20176707,  0.50963248]), array([1, 2, 2, 3])]

>>> print(output)
[-0.45804846 -0.20176707 -0.20176707  0.50963247]
```
---
## Identity ##

**Scala:**
```scala
val identity = Identity()
```
**Python:**
```python
identity = Identity()
```

Identity just return input as the output which is useful in same parallel container to get an origin input

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val identity = Identity()

val input = Tensor(3, 3).rand()
> print(input)
0.043098174	0.1035049	0.7522675	
0.9999951	0.794151	0.18344955	
0.9419861	0.02398399	0.6228095	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]

> print(identity.forward(input))
0.043098174	0.1035049	0.7522675	
0.9999951	0.794151	0.18344955	
0.9419861	0.02398399	0.6228095	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]


```

**Python example:**
```python
from bigdl.nn.layer import *
identity = Identity()
>  identity.forward(np.array([[1, 2, 3], [4, 5, 6]]))
[array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]], dtype=float32)]
       
```
---
## Narrow ##

**Scala:**
```scala
val layer = Narrow(dimension, offset, length = 1)
```
**Python:**
```python
layer = Narrow(dimension, offset, length=1)
```

Narrow is an application of narrow operation in a module.
The module further supports a negative length in order to handle inputs with an unknown size.

Parameters:
* `dimension` narrow along this dimension
* `offset` the start index on the given dimension
* `length` length to narrow, default value is 1

**Scala Example**
```scala
import com.intel.analytics.bigdl.nn.Narrow
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val layer = Narrow(2, 2)
val input = Tensor(T(
  T(-1f, 2f, 3f),
  T(-2f, 3f, 4f),
  T(-3f, 4f, 5f)
))

val gradOutput = Tensor(T(3f, 4f, 5f))

val output = layer.forward(input)
2.0
3.0
4.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x1]

val grad = layer.backward(input, gradOutput)
0.0	3.0	0.0
0.0	4.0	0.0
0.0	5.0	0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python Example**
```python
layer = Narrow(2, 2)
input = np.array([
  [-1.0, 2.0, 3.0],
  [-2.0, 3.0, 4.0],
  [-3.0, 4.0, 5.0]
])

gradOutput = np.array([3.0, 4.0, 5.0])

output = layer.forward(input)
grad = layer.backward(input, gradOutput)

print output
[[ 2.]
 [ 3.]
 [ 4.]]

print grad
[[ 0.  3.  0.]
 [ 0.  4.  0.]
 [ 0.  5.  0.]]
```
---
## Unsqueeze ##

**Scala:**
```scala
val layer = Unsqueeze(dim)
```
**Python:**
```python
layer = Unsqueeze(dim)
```

Insert singleton dim (i.e., dimension 1) at position pos. For an input with `dim = input.dim()`,
there are `dim + 1` possible positions to insert the singleton dimension. The dim starts from 1.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val layer = Unsqueeze(2)
val input = Tensor(2, 2, 2).rand
val gradOutput = Tensor(2, 1, 2, 2).rand
val output = layer.forward(input)
val gradInput = layer.backward(input, gradOutput)

> println(input.size)
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2]

> println(gradOutput.size)
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1x2x2]

> println(output.size)
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1x2x2]

> println(gradInput.size)
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

layer = Unsqueeze(2)
input = np.random.uniform(0, 1, (2, 2, 2)).astype("float32")
gradOutput = np.random.uniform(0, 1, (2, 1, 2, 2)).astype("float32")

output = layer.forward(input)
gradInput = layer.backward(input, gradOutput)

> output
[array([[[[ 0.97488612,  0.43463323],
          [ 0.39069486,  0.0949123 ]]],
 
 
        [[[ 0.19310953,  0.73574477],
          [ 0.95347691,  0.37380624]]]], dtype=float32)]
> gradInput
[array([[[ 0.9995622 ,  0.69787127],
         [ 0.65975296,  0.87002522]],
 
        [[ 0.76349133,  0.96734989],
         [ 0.88068211,  0.07284366]]], dtype=float32)]
```
---
## Squeeze ##

**Scala:**
```scala
val module = Squeeze(dims=null, numInputDims=Int.MinValue)
```
**Python:**
```python
module = Squeeze(dims, numInputDims=-2147483648)
```

Delete all singleton dimensions or a specific singleton dimension.

* `dims` Optional. If this dimension is singleton dimension, it will be deleted.
           The first index starts from 1. Default: delete all dimensions.
* `num_input_dims` Optional. If in a batch model, set to the inputDims.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val layer = Squeeze(2)
> print(layer.forward(Tensor(2, 1, 3).rand()))
0.43709445	0.42752415	0.43069172	
0.67029667	0.95641375	0.28823504	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
from bigdl.nn.layer import *

layer = Squeeze(2)
>layer.forward(np.array([[[1, 2, 3]], [[1, 2, 3]]]))
out: array([[ 1.,  2.,  3.],
            [ 1.,  2.,  3.]], dtype=float32)

```
---
## Select ##

**Scala:**
```scala
val layer = Select(dim, index)
```
**Python:**
```python
layer = Select(dim, index)
```

A Simple layer selecting an index of the input tensor in the given dimension.
Please note that the index and dimension start from 1. In collaborative filtering, it can used together with LookupTable to create embeddings for users or items.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val layer = Select(1, 2)
layer.forward(Tensor(T(
  T(1.0f, 2.0f, 3.0f),
  T(4.0f, 5.0f, 6.0f),
  T(7.0f, 8.0f, 9.0f)
)))

layer.backward(Tensor(T(
  T(1.0f, 2.0f, 3.0f),
  T(4.0f, 5.0f, 6.0f),
  T(7.0f, 8.0f, 9.0f)
)), Tensor(T(0.1f, 0.2f, 0.3f)))
```
Gives the output,
```
4.0
5.0
6.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]

0.0     0.0     0.0
0.1     0.2     0.3
0.0     0.0     0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**
```python
from bigdl.nn.layer import Select
import numpy as np

layer = Select(1, 2)
layer.forward(np.array([
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0],
  [7.0, 8.0, 9.0]
]))
layer.backward(np.array([
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0],
  [7.0, 8.0, 9.0]
]), np.array([0.1, 0.2, 0.3]))
```
Gives the output,
```
array([ 4.,  5.,  6.], dtype=float32)

array([[ 0.        ,  0.        ,  0.        ],
       [ 0.1       ,  0.2       ,  0.30000001],
       [ 0.        ,  0.        ,  0.        ]], dtype=float32)
```
---
## MaskedSelect ##

**Scala:**
```scala
val module = MaskedSelect()
```
**Python:**
```python
module = MaskedSelect()
```

Performs a torch.MaskedSelect on a Tensor. The mask is supplied as a tabular argument
 with the input on the forward and backward passes.
 
**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import scala.util.Random


val layer = MaskedSelect()
val input1 = Tensor(2, 2).apply1(e => Random.nextFloat())
val mask = Tensor(2, 2)
mask(Array(1, 1)) = 1
mask(Array(1, 2)) = 0
mask(Array(2, 1)) = 0
mask(Array(2, 2)) = 1
val input = T()
input(1.0) = input1
input(2.0) = mask
> print(layer.forward(input))
0.2577119
0.5061479
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
```

**Python example:**
```python
from bigdl.nn.layer import *

layer = MaskedSelect()
input1 = np.random.rand(2,2)
mask = np.array([[1,0], [0, 1]])
>layer.forward([input1, mask])
array([ 0.1525335 ,  0.05474588], dtype=float32)
```
---
## Transpose ##

**Scala:**
```scala
val module = Transpose(permutations)
```
**Python:**
```python
module = Transpose(permutations)
```

Concat is a layer who transpose input along specified dimensions.
permutations are dimension pairs that need to swap.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val input = Tensor(2, 3).rand()
val layer = Transpose(Array((1, 2)))
val output = layer.forward(input)

> input
0.6653826	0.25350887	0.33434764	
0.9618287	0.5484164	0.64844745	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

> output
0.6653826	0.9618287	
0.25350887	0.5484164	
0.33434764	0.64844745	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

layer = Transpose([(1,2)])
input = np.array([[0.6653826, 0.25350887, 0.33434764], [0.9618287, 0.5484164, 0.64844745]])
output = layer.forward(input)

> output
[array([[ 0.66538262,  0.96182871],
       [ 0.25350887,  0.54841638],
       [ 0.33434764,  0.64844745]], dtype=float32)]

```
---
## InferReshape ##

**Scala:**
```scala
val layer = InferReshape(size, batchMode = false)
```
**Python:**
```python
layer = InferReshape(size, batch_mode=False)
```

Reshape the input tensor with automatic size inference support.
Positive numbers in the `size` argument are used to reshape the input to the
corresponding dimension size.

There are also two special values allowed in `size`:

   1. `0` means keep the corresponding dimension size of the input unchanged.
      i.e., if the 1st dimension size of the input is 2,
      the 1st dimension size of output will be set as 2 as well.
   2. `-1` means infer this dimension size from other dimensions.
      This dimension size is calculated by keeping the amount of output elements
      consistent with the input.
      Only one `-1` is allowable in `size`.

For example,
```
   Input tensor with size: (4, 5, 6, 7)
   -> InferReshape(Array(4, 0, 3, -1))
   Output tensor with size: (4, 5, 3, 14)
```

The 1st and 3rd dim are set to given sizes, keep the 2nd dim unchanged,
and inferred the last dim as 14.

Parameters:
* `size` the target tensor size
* `batchMode` whether in batch mode

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.InferReshape
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat

val layer = InferReshape(Array(0, 3, -1))
val input = Tensor(1, 2, 3).rand()
val gradOutput = Tensor(1, 3, 2).rand()

val output = layer.forward(input)
val grad = layer.backward(input, gradOutput)

println(output)
(1,.,.) =
0.8170822	0.40073588
0.49389255	0.3782435
0.42660004	0.5917206

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x2]

println(grad)
(1,.,.) =
0.8294597	0.57101834	0.90910035
0.32783163	0.30494633	0.7339092

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x3]
```

**Python example:**
```python
layer = InferReshape([0, 3, -1])
input = np.random.rand(1, 2, 3)

gradOutput = np.random.rand(1, 3, 2)

output = layer.forward(input)
grad = layer.backward(input, gradOutput)

print output
[[[ 0.68635464  0.21277553]
  [ 0.13390459  0.65662414]
  [ 0.1021723   0.92319047]]]

print grad
[[[ 0.84927064  0.55205333  0.25077972]
  [ 0.76105869  0.30828172  0.1237276 ]]]
```
---
## Replicate ##

**Scala:**
```scala
val module = Replicate(
  nFeatures,
  dim = 1,
  nDim = Int.MaxValue)
```
**Python:**
```python
module = Replicate(
  n_features,
  dim=1,
  n_dim=INTMAX)
```
Replicate repeats input `nFeatures` times along its `dim` dimension

Notice: No memory copy, it set the stride along the `dim`-th dimension to zero.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val module = Replicate(4, 1, 2)

println(module.forward(Tensor.range(1, 6, 1).resize(1, 2, 3)))
```
Gives the output,
```
com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
1.0	2.0	3.0
4.0	5.0	6.0

(1,2,.,.) =
1.0	2.0	3.0
4.0	5.0	6.0

(1,3,.,.) =
1.0	2.0	3.0
4.0	5.0	6.0

(1,4,.,.) =
1.0	2.0	3.0
4.0	5.0	6.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x4x2x3]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Replicate(4, 1, 2)

print(module.forward(np.arange(1, 7, 1).reshape(1, 2, 3)))
```
Gives the output, 
```
[array([[[[ 1.,  2.,  3.],
         [ 4.,  5.,  6.]],

        [[ 1.,  2.,  3.],
         [ 4.,  5.,  6.]],

        [[ 1.,  2.,  3.],
         [ 4.,  5.,  6.]],

        [[ 1.,  2.,  3.],
         [ 4.,  5.,  6.]]]], dtype=float32)]
```

## View ##

**Scala:**

```scala
val view = View(2, 8)
```

or

```
val view = View(Array(2, 8))
```

**Python:**
```python
view = View([2, 8])
```

This module creates a new view of the input tensor using the sizes passed to the constructor.
The method setNumInputDims() allows to specify the expected number of dimensions of the inputs
of the modules. This makes it possible to use minibatch inputs
when using a size -1 for one of the dimensions.

**Scala example:**

```scala
import com.intel.analytics.bigdl.nn.View
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val view = View(2, 8)

val input = Tensor(4, 4).randn()
val gradOutput = Tensor(2, 8).randn()

val output = view.forward(input)
val gradInput = view.backward(input, gradOutput)
```
Gives the output,
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.43037438     1.2982363       -1.4723133      -0.2602826      0.7178128       -1.8763185      0.88629466      0.8346704
0.20963766      -0.9349786      1.0376515       1.3153045       1.5450214       1.084113        -0.29929757     -0.18356979
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x8]
```
Gives the gradInput,
```
gradInput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.7360089       0.9133299       0.40443268      -0.94965595
0.80520976      -0.09671917     -0.5498001      -0.098691925
-2.3119886      -0.8455147      0.75891125      1.2985301
0.5023749       1.4983269       0.42038065      -1.7002305
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

view = View([2, 8])

input = np.random.uniform(0, 1, [4, 4]).astype("float32")
gradOutput = np.random.uniform(0, 1, [2, 8]).astype("float32")

output = view.forward(input)
gradInput = view.backward(input, gradOutput)

print output
print gradInput
```
---
## Contiguous ##

Be used to make input, gradOutput both contiguous

**Scala:**
```scala
val contiguous = Contiguous()
```

**Python:**
```python
contiguous = Contiguous()
```
Used to make input, gradOutput both contiguous

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Contiguous
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val input = Tensor(5).range(1, 5, 1)
val contiguous = new Contiguous()
val output = contiguous.forward(input)
println(output)

val gradOutput = Tensor(5).range(2, 6, 1)
val gradInput = contiguous.backward(input, gradOutput)
println(gradOutput)
```
Gives the output,
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0
2.0
3.0
4.0
5.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]
```
Gives the gradInput,

```
gradInput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
2.0
3.0
4.0
5.0
6.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

contiguous = Contiguous()

input = np.arange(1, 6, 1).astype("float32")
input = input.reshape(1, 5)

output = contiguous.forward(input)
print output

gradOutput = np.arange(2, 7, 1).astype("float32")
gradOutput = gradOutput.reshape(1, 5)

gradInput = contiguous.backward(input, gradOutput)
print gradInput

```
Gives the output,
```
[array([[ 1.,  2.,  3.,  4.,  5.]], dtype=float32)]
```

Gives the gradInput,

```
[array([[ 2.,  3.,  4.,  5.,  6.]], dtype=float32)]
```

## GaussianSampler ##

Takes {mean, log_variance} as input and samples from the Gaussian distribution

**Scala:**
```scala
val sampler = GaussianSampler()
```

**Python:**
```python
sampler = GaussianSampler()
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.GaussianSampler
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T

val input1 = Tensor[Float](2, 3).range(1, 6, 1)
val input2 = Tensor[Float](2, 3).range(1, 12, 2)
val input = T(input1, input2)

val gradOutput = Tensor[Float](2, 3).range(2, 13, 2)
    
val sampler = new GaussianSampler()
val output = sampler.forward(input)
println(output)

val gradInput = sampler.backward(input, gradOutput)
println(gradOutput)
```
Gives the output,
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
4.507061	9.247583	-14.053247	
34.783264	-70.69336	-333.97656	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]
```
Gives the gradInput,

```
gradInput: com.intel.analytics.bigdl.utils.Table = 
 {
	1: 2.0	4.0     6.0	
	   8.0	10.0	12.0	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
    2: 3.5070612	14.495168	-51.159744	
       123.13305	-378.4668	-2039.8594	
       [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
 }
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
import numpy as np

sampler = GaussianSampler()

input1 = np.arange(1, 7, 1).astype("float32")
input2 = np.arange(1, 12, 2).astype("float32")
input2 = input1.reshape(2, 3)
input2 = input2.reshape(2, 3)
input = [input1, input2]

gradOutput = np.arange(2, 13, 2).astype("float32")
gradOutput = gradOutput.reshape(2, 3)

output = sampler.forward(input)
gradInput = sampler.backward(input, gradOutput)

```
Gives the output,
```
>>> print output
[[ 1.73362803  2.99371576  0.44359136]
 [ 0.04700017  2.85183263  3.04418468]]
```

Gives the gradInput,

```
>>> print gradInput
[array([[  2.,   4.,   6.],
       [  8.,  10.,  12.]], dtype=float32), array([[  0.73362803,   1.98743176,  -7.66922569],
       [-15.81199932, -10.7408371 , -17.73489189]], dtype=float32)]
```

## Masking ##

Use a mask value to skip timesteps for a sequence

**Scala:**
```scala
val mask = Masking(0.0)
```

**Python:**
```python
mask = Masking(0.0)
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Masking
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val batchSize = 3
val times = 5
val features = 2
val inputData = Array[Double](1.0, 1, 2, 2, 3, 3, 4, 4, 5, 5, -1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
  1, 1, -1, -1, 3, 3, 4, 4, 5, 5)
val input = Tensor[Double](inputData, Array(batchSize, times, features))
val gradOutput = Tensor[Double](Array(batchSize, times, features)).fill(1.0)
val maskValue = -1

val mask = Masking(maskValue)
val output = mask.forward(input)
println(output)

val gradInput = mask.backward(input, gradOutput)
println(gradOutput)
```
Gives the output,
```
output: = 
(1,.,.) =
1.0	1.0	
2.0	2.0	
3.0	3.0	
4.0	4.0	
5.0	5.0	

(2,.,.) =
-1.0	1.0	
2.0	2.0	
3.0	3.0	
4.0	4.0	
5.0	5.0	

(3,.,.) =
1.0	1.0	
0.0	0.0	
3.0	3.0	
4.0	4.0	
5.0	5.0	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x5x2]
```
Gives the gradInput,

```
gradInput: 
(1,.,.) =
1.0	1.0	
1.0	1.0	
1.0	1.0	
1.0	1.0	
1.0	1.0	

(2,.,.) =
1.0	1.0	
1.0	1.0	
1.0	1.0	
1.0	1.0	
1.0	1.0	

(3,.,.) =
1.0	1.0	
0.0	0.0	
1.0	1.0	
1.0	1.0	
1.0	1.0	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x5x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.util.common import *
import numpy as np

n_samples = 3
n_timesteps = 7
n_features = 2
mask_value = -1.0
input = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, -1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                 1, 1, 2, 2, 3, 3, -1, -1, 5, 5, 6, 6, 7, 7]).reshape(n_samples, n_timesteps, n_features)
gradOutput = np.ones((n_samples, n_timesteps, n_features))
model = Sequential()
model.add(Masking(mask_value=mask_value))

output = model.forward(input)
gradInput = model.backward(input, gradOutput)

```
Gives the output,
```
>>> print output
[[[ 1.  1.]
  [ 2.  2.]
  [ 3.  3.]
  [ 4.  4.]
  [ 5.  5.]
  [ 6.  6.]
  [ 7.  7.]]

 [[-1.  1.]
  [ 2.  2.]
  [ 3.  3.]
  [ 4.  4.]
  [ 5.  5.]
  [ 6.  6.]
  [ 7.  7.]]

 [[ 1.  1.]
  [ 2.  2.]
  [ 3.  3.]
  [ 0.  0.]
  [ 5.  5.]
  [ 6.  6.]
  [ 7.  7.]]]
```

Gives the gradInput,

```
>>> print gradInput
[[[ 1.  1.]
  [ 1.  1.]
  [ 1.  1.]
  [ 1.  1.]
  [ 1.  1.]
  [ 1.  1.]
  [ 1.  1.]]

 [[ 1.  1.]
  [ 1.  1.]
  [ 1.  1.]
  [ 1.  1.]
  [ 1.  1.]
  [ 1.  1.]
  [ 1.  1.]]

 [[ 1.  1.]
  [ 1.  1.]
  [ 1.  1.]
  [ 0.  0.]
  [ 1.  1.]
  [ 1.  1.]
  [ 1.  1.]]]
```

## Maxout ##    

**Scala:**
```scala
val maxout = Maxout(2, 5, 3,
                    withBias = true,
                    wRegularizer = null,
                    bRegularizer = null,
                    initWeight = null,
                    initBias = null)
```

**Python:**
```python
maxout = Maxout(2, 5, 3,
                 with_bias = True,
                 w_regularizer=None,
                 b_regularizer=None,
                 init_weight=None,
                 init_bias=None)
```

Maxout layer select the element-wise maximum value of maxoutNumber Linear(inputSize, outputSize) layers

parameters:
* `inputSize` the size the each input sample
* `outputSize` the size of the module output of each sample
* `maxoutNumber` number of Linear layers to use
* `withBias` whether use bias in Linear
* `wRegularizer` instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the input weights matrices.
* `bRegularizer` instance of [[Regularizer]] applied to the bias.
* `initWeight` initial weight
* `initBias` initial bias

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Maxout
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val input_size = 2
val batch_size = 3
val output_size = 5
val maxout_number = 3

val input = Tensor[Float](batch_size, input_size).rand()
val layer = Maxout[Float](input_size, output_size, maxout_number)
val output = layer.forward(input)
val gradOutput = Tensor[Float](batch_size, output_size)
val gradInput = layer.backward(input, gradOutput)
```
Gives the output,
```
0.19078568	0.94480306	0.25038794	0.8114594	0.7753764	
0.2822805	0.9095781	0.2815394	0.82958585	0.784589	
0.35188058	0.7629706	0.18096384	0.7100433	0.6680352	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x5]
```
Gives the gradInput,

```
gradInput: 
-0.18932924	0.9426162	
-0.3118648	0.67255044	
-0.31795382	1.944398	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.util.common import *
import numpy as np

val input_size = 2
val batch_size = 3
val output_size = 5
val maxout_number = 3

val input = Tensor[Float](batch_size, input_size).rand()
val layer = Maxout[Float](input_size, output_size, maxout_number)
val output = layer.forward(input)
val gradOutput = Tensor[Float](batch_size, output_size).rand()
val gradInput = layer.backward(input, gradOutput)

```
Gives the output,
```
>>> print output
[[ 0.12344513  0.19081372  0.15130989  0.6341747   0.70982581]
 [ 0.04154952 -0.13281995  0.2648508   0.36793122  0.67043799]
 [ 0.41355255  0.17691913  0.15496807  0.5880245   0.74583203]]
```

Gives the gradInput,

```
>>> print gradInput
[[ 0.53398496  0.01809531]
 [-0.20667852  0.4962275 ]
 [ 0.37912956  0.08742841]]
```

## Cropping2D ##

**Scala:**
```scala
val module = Cropping2D(heightCrop, widthCrop, dataFormat=DataFormat.NCHW)
```
**Python:**
```python
m = Cropping2D(heightCrop, widthCrop, data_format="NCHW")
```

Cropping layer for 2D input (e.g. picture). It crops along spatial dimensions, i.e. width and height.
    # Arguments
        heightCrop: Array of length 2. How many units should be trimmed off at the
                    beginning and end of the height dimension.
        widthCrop: Array of length 2. How many units should be trimmed off at the
                   beginning and end of the width dimension
        dataFormat: DataFormat.NCHW or DataFormat.NHWC.
    # Input shape
        4D tensor with shape:
        `(samples, depth, first_axis_to_crop, second_axis_to_crop)`
    # Output shape
        4D tensor with shape:
        `(samples, depth, first_cropped_axis, second_cropped_axis)`

**Scala example:**
```scala

scala >
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.Storage

val module = Cropping2D(Array(1, 1), Array(1, 1))
val input = Tensor(2, 1, 3, 3).rand()
val output = module.forward(input)

> input
(1,1,.,.) =
0.024445634	0.73160243	0.1408418	
0.95527077	0.51474196	0.89850646	
0.3730063	0.40874788	0.7043526	

(2,1,.,.) =
0.8549189	0.5019415	0.96255547	
0.83960533	0.3738476	0.12785637	
0.08048103	0.6209139	0.6762928	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x1x3x3]

> output
(1,1,.,.) =
0.51474196	

(2,1,.,.) =
0.3738476	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1x1x1]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input = np.random.rand(2, 1, 3, 3)
print ("input is :",input)

m = Cropping2D([1, 1], [1, 1])
out = m.forward(input)
print ("output m is :",out)
```
Gives the output,

```python
input is : [[[[ 0.54167415  0.59110695  0.583436  ]
   [ 0.7401184   0.93070248  0.88311626]
   [ 0.08472445  0.90583803  0.83751593]]]


 [[[ 0.98047837  0.13156681  0.73104089]
   [ 0.15081809  0.1791556   0.18849927]
   [ 0.12054713  0.75931796  0.40090047]]]]
creating: createCropping2D
output m is : [[[[ 0.93070251]]]


 [[[ 0.1791556 ]]]]
```

## Cropping3D ##

**Scala:**
```scala
val module = Cropping3D(dim1Crop, dim2Crop, dim3Crop, dataFormat="channel_first")
```
**Python:**
```python
m = Cropping3D(dim1Crop, dim2Crop, dim3Crop, dataFormat="channel_first")
```

Cropping layer for 3D data (e.g. spatial or spatio-temporal).
    # Arguments
        dim1Crop, dim2Crop, dim3Crop: each is an Array of two int, specifies how
                                      many units should be trimmed off at the
                                      beginning and end of the 3 cropping dimensions.
        dataFormat: Cropping3D.CHANNEL_FIRST or Cropping3D.CHANNEL_LAST

**Scala example:**
```scala

scala >
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.Storage

val module = Cropping3D(Array(1, 1), Array(1, 1), Array(1, 1))
val input = Tensor(2, 1, 3, 3, 3).rand()
val output = module.forward(input)

> input
(1,1,1,.,.) =
0.33822843	0.83652526	0.6983564	
0.40552914	0.50253755	0.26770833	
0.12843947	0.7388038	0.8611642	

(1,1,2,.,.) =
0.52169484	0.98340595	0.37585744	
0.47124776	0.1858571	0.20025288	
0.24735944	0.68807006	0.12379094	

(1,1,3,.,.) =
0.3149784	0.43712634	0.9625379	
0.37466723	0.8551855	0.7831635	
0.979082	0.6115703	0.09862939	

(2,1,1,.,.) =
0.8603551	0.64941335	0.382916	
0.9402129	0.83625364	0.41554055	
0.9974375	0.7845985	0.4631692	

(2,1,2,.,.) =
0.41448194	0.06975327	0.68035746	
0.6495608	0.95513606	0.5103921	
0.4187052	0.676009	0.00466285	

(2,1,3,.,.) =
0.043842442	0.9419528	0.9560404	
0.8702963	0.4117603	0.91820705	
0.39294028	0.010171742	0.23027366	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x1x3x3x3]

> output
(1,1,1,.,.) =
0.1858571	

(2,1,1,.,.) =
0.95513606	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1x1x1x1]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input = np.random.rand(2, 1, 3, 3, 3)
print ("input is :",input)

m = Cropping3D([1, 1], [1, 1], [1, 1])
out = m.forward(input)
print ("output m is :",out)
```
Gives the output,

```python
input is : [[[[[ 0.00484727  0.64335228  0.21672991]
    [ 0.6667991   0.90280284  0.17537352]
    [ 0.17573056  0.51962225  0.7946977 ]]

   [[ 0.54374072  0.02084648  0.817017  ]
    [ 0.10707117  0.96247797  0.97634706]
    [ 0.23012049  0.7498735   0.67309293]]

   [[ 0.22704888  0.31254715  0.59703825]
    [ 0.61084924  0.55686219  0.55321829]
    [ 0.75911533  0.00731942  0.20643018]]]]



 [[[[ 0.89015703  0.28932907  0.80356569]
    [ 0.55100695  0.66712567  0.00770912]
    [ 0.91482596  0.43556021  0.96402856]]

   [[ 0.36694364  0.27634374  0.52885899]
    [ 0.40754185  0.79033726  0.42423772]
    [ 0.20636923  0.72467024  0.80372414]]

   [[ 0.50318154  0.54954067  0.71939314]
    [ 0.52834256  0.26762247  0.32269808]
    [ 0.53824181  0.42523858  0.95246198]]]]]
creating: createCropping3D
output m is : [[[[[ 0.96247798]]]]



 [[[[ 0.79033726]]]]]
```