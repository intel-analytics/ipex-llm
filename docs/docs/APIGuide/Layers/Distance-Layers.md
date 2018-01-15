## PairwiseDistance ##

**Scala:**
```scala
val pd = PairwiseDistance(norm=2)
```
**Python:**
```python
pd = PairwiseDistance(norm=2)
```

It is a module that takes a table of two vectors as input and outputs
the distance between them using the p-norm.
The input given in `forward(input)` is a [[Table]] that contains two tensors which
must be either a vector (1D tensor) or matrix (2D tensor). If the input is a vector,
it must have the size of `inputSize`. If it is a matrix, then each row is assumed to be
an input sample of the given batch (the number of rows means the batch size and
the number of columns should be equal to the `inputSize`).

**Scala example:**

```scala
import com.intel.analytics.bigdl.nn.PairwiseDistance
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val pd = PairwiseDistance()
val input1 = Tensor(3, 3).randn()
val input2 = Tensor(3, 3).randn()
val input = T(1 -> input1, 2 -> input2)

val output = pd.forward(input)

val gradOutput = Tensor(3).randn()
val gradInput = pd.backward(input, gradOutput)

```

The ouotput is,

```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
4.155246
1.1267666
2.1415536
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
gradOutput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.32565984
-1.0108998
-0.030873261
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
```

The gradInput is,

```
gradInput: com.intel.analytics.bigdl.utils.Table =
 {
        2: 0.012723052  0.31482473      0.08232752
           0.7552968    -0.27292773     -0.6139655
           0.0062761847 -0.018232936    -0.024110721
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
        1: -0.012723052 -0.31482473     -0.08232752
           -0.7552968   0.27292773      0.6139655
           -0.0062761847        0.018232936     0.024110721
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
 }

```

**Python example:**

```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

pd = PairwiseDistance()

input1 = np.random.uniform(0, 1, [3, 3]).astype("float32")
input2 = np.random.uniform(0, 1, [3, 3]).astype("float32")
input1 = input1.reshape(3, 3)
input2 = input2.reshape(3, 3)

input = [input1, input2]

output = pd.forward(input)
print output

gradOutput = np.random.uniform(0, 1, [3]).astype("float32")
gradOutput = gradOutput.reshape(3)

gradInput = pd.backward(input, gradOutput)
print gradInput
```

The output is,

```
[ 0.99588805  0.65620303  1.11735415]
```

The gradInput is,

```
[array([[-0.27412388,  0.32756016, -0.02032043],
       [-0.16920818,  0.60189474,  0.21347123],
       [ 0.57771122,  0.28602061,  0.58044904]], dtype=float32), array([[ 0.27412388, -0.32756016,  0.02032043],
       [ 0.16920818, -0.60189474, -0.21347123],
       [-0.57771122, -0.28602061, -0.58044904]], dtype=float32)]
```

## CosineDistance ##

**Scala:**
```scala
val module = CosineDistance()
```
**Python:**
```python
module = CosineDistance()
```

CosineDistance creates a module that takes a table of two vectors (or matrices if in batch mode) as input and outputs the cosine distance between them.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val module = CosineDistance()
val t1 = Tensor().range(1, 3)
val t2 = Tensor().range(4, 6)
val input = T(t1, t2)
val output = module.forward(input)

> input
input: com.intel.analytics.bigdl.utils.Table =
 {
	2: 4.0
	   5.0
	   6.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
	1: 1.0
	   2.0
	   3.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
 }

> output
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.9746319
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = CosineDistance()
t1 = np.array([1.0, 2.0, 3.0])
t2 = np.array([4.0, 5.0, 6.0])
input = [t1, t2]
output = module.forward(input)

> input
[array([ 1.,  2.,  3.]), array([ 4.,  5.,  6.])]

> output
[ 0.97463191]
```

## Euclidean ##

**Scala:**
```scala
val module = Euclidean(
  inputSize,
  outputSize,
  fastBackward = true)
```
**Python:**
```python
module = Euclidean(
  input_size,
  output_size,
  fast_backward=True)
```
Outputs the Euclidean distance of the input to `outputSize` centers.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val module = Euclidean(3, 3)

println(module.forward(Tensor.range(1, 3, 1)))
```
Output is
```
com.intel.analytics.bigdl.tensor.Tensor[Float] =
4.0323668
3.7177157
3.8736997
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Euclidean(3, 3)

print(module.forward(np.arange(1, 4, 1)))
```
Output is
```
[array([ 3.86203027,  4.02212906,  3.2648952 ], dtype=float32)]
```

