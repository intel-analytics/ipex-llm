## MapTable ##

**Scala:**
```scala
val mod = MapTable(module=null)
```
**Python:**
```python
mod = MapTable(module=None)
```

This class is a container for a single module which will be applied
to all input elements. The member module is cloned as necessary to
process all input elements.

`module` a member module.  
 
**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T 

val map = MapTable()
map.add(Linear(10, 3))
val input = T(
      Tensor(10).randn(),
      Tensor(10).randn())
> print(map.forward(input))
{
	2: 0.2444828
	   -1.1700082
	   0.15887381
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
	1: 0.06696482
	   0.18692614
	   -1.432079
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
 }
```

**Python example:**
```python
from bigdl.nn.layer import *

map = MapTable()
map.add(Linear(10, 3))
input = [np.random.rand(10), np.random.rand(10)]
>map.forward(input)
[array([ 0.69586945, -0.70547599, -0.05802459], dtype=float32),
 array([ 0.47995114, -0.67459631, -0.52500772], dtype=float32)]
```