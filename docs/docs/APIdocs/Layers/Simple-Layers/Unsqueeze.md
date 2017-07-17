## Unsqueeze ##

**Scala:**
```scala
val layer = Unsqueeze(dim)
```
**Python:**
```python
layer = Unsqueeze(dim)
```

Insert singleton dim (i.e., dimension 1) at position pos. For an input with dim = input.dim(),
there are dim + 1 possible positions to insert the singleton dimension. The dim starts from 1.

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
