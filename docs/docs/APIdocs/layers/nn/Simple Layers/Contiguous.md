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

Description

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

The output will be,

```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0
2.0
3.0
4.0
5.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]
```

The gradInput will be,

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

The output will be,

```
[array([[ 1.,  2.,  3.,  4.,  5.]], dtype=float32)]
```

The gradInput will be,

```
[array([[ 2.,  3.,  4.,  5.,  6.]], dtype=float32)]
```