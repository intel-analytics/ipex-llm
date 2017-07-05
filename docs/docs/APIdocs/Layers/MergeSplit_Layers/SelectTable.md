## SelectTable ##

**Scala:**
```scala
val m = SelectTable(index: Int)
```
**Python:**
```python
m = SelectTable(dimension)
```

Select one element from a table by a given index.
In Scala API, table is kind of like HashMap with one-base index as the key.
In python, table is a just a list.


**Scala example:**

```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val input = T(Tensor(2,3).randn(), Tensor(2,3).randn())

println("input: ")
println(input)
println("output:")
println(SelectTable(1).forward(input)) // Select and output the first element of the input which shape is (2, 3)
println(SelectTable(2).forward(input)) // Select and output the second element of the input which shape is (2, 3)

```
```
input: 
 {
	2: 2.005436370849835	0.09670211785545313	1.186779895312918	
	   2.238415300857082	0.241626512721254	0.15765709974113828	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
	1: 0.5668905654052705	-1.3205159007397167	-0.5431464848526197	
	   -0.11582559521074104	0.7671830693813515	-0.39992781407893574	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
 }
output:
0.5668905654052705	-1.3205159007397167	-0.5431464848526197	
-0.11582559521074104	0.7671830693813515	-0.39992781407893574	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
2.005436370849835	0.09670211785545313	1.186779895312918	
2.238415300857082	0.241626512721254	0.15765709974113828	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]

```

**Python example:**
```python
import numpy as np
from bigdl.nn.layer import *

input = [np.random.random((2,3)), np.random.random((2, 1))]
print("input:")
print(input)
print("output:")
print(SelectTable(1).forward(input)) # Select and output the first element of the input which shape is (2, 3)
```
```
input:
[array([[ 0.07185111,  0.26140439,  0.9437582 ],
       [ 0.50278191,  0.83923974,  0.06396735]]), array([[ 0.84955122],
       [ 0.16053703]])]
output:
creating: createSelectTable
[[ 0.07185111  0.2614044   0.94375819]
 [ 0.50278193  0.83923972  0.06396735]]
 
```

