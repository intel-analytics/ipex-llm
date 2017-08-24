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

def randomn(): Float = RandomGenerator.RNG.uniform(0, 1)
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
* `batchMode` It is a optional argument. If it is set to `Some(true)`,
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
