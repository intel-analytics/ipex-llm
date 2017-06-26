## HardShrink ##


**Scala:**
```scala
HardShrink[T](lambda: Double = 0.5)
```
**Python:**
```python
HardShrink(the_lambda=0.5, bigdl_type="float")
```

Applies the hard shrinkage function element-wise to the input Tensor. lambda is set to 0.5 by default.

HardShrinkage operator is defined as:

```
       ⎧ x, if x >  lambda
f(x) = ⎨ x, if x < -lambda
       ⎩ 0, otherwise
```



**Scala example:**
```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._


import com.intel.analytics.bigdl.utils._

def randomn(): Double = RandomGenerator.RNG.uniform(-10, 10)
val input = Tensor[Double](3, 4)
input.apply1(x => randomn())

val layer = new HardShrink[Double](8)
println("input:")
println(input)
println("output:")
println(layer.forward(input))
```

```
input:
8.53746839798987	-2.25314284209162	2.838596091605723	0.7181660132482648	
0.8278933027759194	8.986027473583817	-3.6885232804343104	-2.4018199276179075	
-9.51015486381948	2.6402589259669185	5.438693333417177	-6.577442386187613	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4]
output:
8.53746839798987	0.0	0.0	0.0	
0.0	8.986027473583817	0.0	0.0	
-9.51015486381948	0.0	0.0	0.0	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4]
```

**Python example:**


```python
import numpy as np
from bigdl.nn.layer import *

input = np.linspace(-5, 5, num=10)
layer = HardShrink(the_lambda=3.0)
print("input:")
print(input)
print("output: ")
print(layer.forward(input))
```

```
creating: createHardShrink
input:
[-5.         -3.88888889 -2.77777778 -1.66666667 -0.55555556  0.55555556
  1.66666667  2.77777778  3.88888889  5.        ]
output: 
[-5.         -3.88888884  0.          0.          0.          0.          0.
  0.          3.88888884  5.        ]
 
```

