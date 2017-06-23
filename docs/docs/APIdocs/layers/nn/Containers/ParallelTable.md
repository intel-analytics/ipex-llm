## ParallelTable ##

**Scala:**
```scala
val module = ParallelTable()
```
**Python:**
```python
module = ParallelTable()
```

It is a container module that applies the i-th member module to the i-th
 input, and outputs an output in the form of Table
 
**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T 

val module = ParallelTable[Double]()
val log = Log[Double]()
val exp = Exp[Double]()
module.add(log)
module.add(exp)
val input1 = Tensor[Double](3, 3).rand(0, 1)
val input2 = Tensor[Double](3).rand(0, 1)
val input = T(1 -> input1, 2 -> input2)
> print(module.forward(input))
{
	2: 1.446654561455216
	   2.583952650388725
	   2.0971580389678044
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
	1: -1.1356028994540759	-0.2522996245342669	-0.25814063641058016	
	   -0.2379749641235881	-1.2113681814751194	-3.5294739520242406	
	   -0.6222607227092349	-1.0116901818257722	-0.30867721346604926	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
 }
```

**Python example:**
```python
from bigdl.nn.layer import *

module = ParallelTable()
log = Log()
exp = Exp()
module.add(log)
module.add(exp)
input1 = np.random.rand(3,3)
input2 = np.random.rand(3)
>module.forward([input1, input2])
[array([[-1.27472472, -2.18216252, -0.60752904],
        [-2.76213861, -1.77966928, -0.13652121],
        [-1.47725129, -0.03578046, -1.37736678]], dtype=float32),
 array([ 1.10634041,  1.46384597,  1.96525407], dtype=float32)]
```