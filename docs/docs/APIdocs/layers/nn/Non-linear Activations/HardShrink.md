## HardShrink ##

Applies the hard shrinkage function element-wise to the input Tensor. lambda is set to 0.5 by default.

HardShrinkage operator is defined as:

```lua
       ⎧ x, if x >  lambda
f(x) = ⎨ x, if x < -lambda
       ⎩ 0, otherwise
```

**Scala:**
```scala
val layer = new HardShrink[Double](5)
```
**Python:**
```python
HardShrink(the_lambda=0.5, bigdl_type="float")
```


**Scala example:**
```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._


def randomn(): Double = RandomGenerator.RNG.uniform(-10, 10)
val input = Tensor[Double](3, 4)
input.apply1(x => randomn())

val layer = new HardShrink[Double](8)
print(input)
print(layer.forward(input))
```

```
8.870212347246706	2.864008233882487	9.835944692604244	3.2311211759224534	
-7.246359097771347	6.885241577401757	7.250537765212357	7.199446959421039	
-5.146464761346579	5.610057436861098	9.192849891260266	-6.4514826238155365

8.870212347246706  0.0	9.835944692604244 0.0
0.0	               0.0	0.0	              0.0	
0.0	               0.0	9.192849891260266 0.0
```

**Python example:**


```python
import numpy as np
from bigdl.nn.layer import *

input = np.linspace(-5, 5, num=10)
layer = HardShrink(the_lambda=3.0)
print(input)
pred = layer.forward(input)
print(pred)
```

```
output is:
[-5.         -3.88888889 -2.77777778 -1.66666667 -0.55555556  0.55555556
  1.66666667  2.77777778  3.88888889  5.        ]
  
[-5.        , -3.88888884,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  3.88888884,  5.]
 
```

