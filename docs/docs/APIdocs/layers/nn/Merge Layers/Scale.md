## Scale ##


**Scala:**

```scala
Scale[T](Array(2, 1))
```
**Python:**
```python
scale = Scale([2, 1])
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

val input = Tensor[Float](2, 3).fill(1)
println("input:")
println(input)
val scale = Scale[Float](Array(2, 1))
val weight = Tensor[Float](2, 1).fill(2)
val bias = Tensor[Float](2, 1).fill(3)
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

