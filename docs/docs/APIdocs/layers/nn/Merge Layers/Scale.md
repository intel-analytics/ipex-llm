## Scale ##
Scale is the combination of cmul and cadd. `Scale(size).forward(input) == CAdd(size).forward(CMul(size).forward(input))`
Computes the elementwise product of input and weight, with the shape of the weight "expand" to
match the shape of the input.Similarly, perform a expand cdd bias and perform an elementwise add.
`output = input .* weight .+ bias (element wise)`

**Scala:**

```scala
Scale[T](Array(2, 1))
```
**Python:**
```python
scale = Scale([2, 1])
```


**Scala example:**
```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.{T, Table}

val input = Tensor[Double](2, 3).fill(1)
val scale = Scale[Double](Array(2, 1))
scale.setWeightsBias(Array(Tensor[Double](2, 1).fill(2),  Tensor[Double](2, 1).fill(3)))
print(scale.forward(input))
```
```
5.0	5.0	5.0	
5.0	5.0	5.0	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**

```python
import numpy as np
from bigdl.nn.layer import *
input = np.ones([2, 3])
scale = Scale([2, 1])
scale.set_weights([np.full([2, 1], 2), np.full([2, 1], 3)]) # this first element is for weight and the second one is bias
print(scale.forward(input))

```
```
output is:
[array([[ 5.,  5.,  5.],
       [ 5.,  5.,  5.]], dtype=float32)]
```

