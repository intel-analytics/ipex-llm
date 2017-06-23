## SelectTable ##
Select one element from a table by a given index.
In Scala API, table is kind of like HashMap with one-base index as the key.
In python, table is a just a list.

```

**Scala:**
```scala
SelectTable[T](index: Int))
```
**Python:**
```python
SelectTable(self,
                            dimension, bigdl_type="float")
```


**Scala example:**

```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.{T, Table}

val input = T(Tensor[Double](2,3).randn(), Tensor[Double](2,1).randn())

print(SelectTable[Double](1).forward(input)) // Select and output the first element of the input which shape is (2, 3)
print(SelectTable[Double](2).forward(input)) // Select and output the second element of the input which shape is (2, 1)
```
```
Output is:

-0.8189912055912036	1.4082322648121768	0.6907710513451537	
-0.9572757353407211	1.4514950010163312	-1.7835791174017501	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]

-0.7674748725130892	-1.3223796388582973	-1.2099097699510337	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.layer import *

input = [np.random.random((2,3)), np.random.random((2, 1))]
print(SelectTable(1).forward(input)) # Select and output the first element of the input which shape is (2, 3)
print(SelectTable(2).forward(input)) # Select and output the second element of the input which shape is (2, 1)
```
```
output is:
[array([[ 0.86908513,  0.01273194,  0.47717375],
       [ 0.46323711,  0.70621955,  0.3182328 ]], dtype=float32)]

[array([[ 0.8029139 ],
       [ 0.73379868]], dtype=float32)]
 
```

