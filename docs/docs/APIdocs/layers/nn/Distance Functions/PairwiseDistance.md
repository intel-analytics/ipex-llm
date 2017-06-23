## PairwiseDistance ##

**Scala:**
```scala
val pd = PairwiseDistance()
```
**Python:**
```python
pd = PairwiseDistance()
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