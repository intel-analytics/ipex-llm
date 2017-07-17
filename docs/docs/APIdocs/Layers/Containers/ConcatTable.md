## ConcatTable ##

**Scala:**
```scala
val module = ConcatTable()
```
**Python:**
```python
module = ConcatTable()
```

ConcateTable is a container module like Concate. Applies an input
to each member module, input can be a tensor or a table.

ConcateTable usually works with CAddTable and CMulTable to
 implement element wise add/multiply on outputs of two modules.

```
                   +-----------+
             +----> {member1, |
+-------+    |    |           |
| input +----+---->  member2, |
+-------+    |    |           |
   or        +---->  member3} |
 {input}          +-----------+
 
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val mlp = ConcatTable()
mlp.add(Linear(3, 2))
mlp.add(Linear(3, 4))

> print(mlp.forward(Tensor(2, 3).rand()))

{
	2: -0.37111914	0.8542446	-0.362602	-0.75522065	
	   -0.28713673	0.6021913	-0.16525984	-0.44689763	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]
	1: -0.79941726	0.8303885	
	   -0.8342782	0.89961016	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
 }

```

**Python example:**
```python
from bigdl.nn.layer import *

mlp = ConcatTable()
mlp.add(Linear(3, 2))   
mlp.add(Linear(3, 4))
> mlp.forward(np.array([[1, 2, 3], [1, 2, 3]]))
out: [array([[ 1.16408789, -0.1508013 ],
             [ 1.16408789, -0.1508013 ]], dtype=float32),
      array([[-0.24672163, -0.56958938, -0.51390374,  0.64546645],
             [-0.24672163, -0.56958938, -0.51390374,  0.64546645]], dtype=float32)]

```
