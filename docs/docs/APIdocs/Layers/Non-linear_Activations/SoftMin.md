## SoftMin ##

**Scala:**
```scala
val sm = SoftMin()
```
**Python:**
```python
sm = SoftMin()
```

Applies the SoftMin function to an n-dimensional input Tensor, rescaling them so that the
elements of the n-dimensional output Tensor lie in the range (0,1) and sum to 1.
Softmin is defined as: f_i(x) = exp(-x_i - shift) / sum_j exp(-x_j - shift)
where shift = max_i(-x_i).

**Scala example:**

```scala
import com.intel.analytics.bigdl.nn.SoftMin
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val sm = SoftMin()
val input = Tensor(3, 3).range(1, 3 * 3)

val output = sm.forward(input)

val gradOutput = Tensor(3, 3).range(1, 3 * 3).apply1(x => (x / 10.0).toFloat)
val gradInput = sm.backward(input, gradOutput)

```

The output will be,

```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.66524094      0.24472848      0.09003057
0.66524094      0.24472848      0.09003057
0.66524094      0.24472848      0.09003057
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

The gradInput will be,

```
gradInput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.02825874      -0.014077038    -0.014181711
0.028258756     -0.01407703     -0.01418171
0.028258756     -0.014077038    -0.014181707
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**


```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

sm = SoftMin()

input = np.arange(1, 10, 1).astype("float32")
input = input.reshape(3, 3)

output = sm.forward(input)
print output

gradOutput = np.arange(1, 10, 1).astype("float32")
gradOutput = np.vectorize(lambda t: t / 10)(gradOutput)
gradOutput = gradOutput.reshape(3, 3)

gradInput = sm.backward(input, gradOutput)
print gradInput

```
