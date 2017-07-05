## Ones ##


**Scala:**
``` scala
val initMethod = Ones

```
**Python:**
```python
init_method = Ones()
```

Initialization method that set tensor to be ones.



**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val weightInitMethod = Ones
val biasInitMethod = Ones
val model = Linear(3, 2).setName("linear1")
model.setInitMethod(weightInitMethod, biasInitMethod)
println(model.getParametersTable().get("linear1").get)
```

```
 {
	weight: 1.0	1.0	1.0	
	        1.0	1.0	1.0	
	        [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
	bias: 1.0
	      1.0
	      [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
	gradBias: 0.0
	          0.0
	          [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
	gradWeight: 0.0	0.0	0.0	
	            0.0	0.0	0.0	
	            [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
 }


```

**Python example:**
```python
from bigdl.nn.initialization_method import *
weight_init = Ones()
bias_init = Ones()
model = Linear(3, 2)
model.set_init_method(weight_init, bias_init)
print("weight:")
print(model.get_weights()[0])
print("bias: ")
print(model.get_weights()[1])
```
```
creating: createOnes
creating: createOnes
creating: createLinear
weight:
[[ 1.  1.  1.]
 [ 1.  1.  1.]]
bias: 
[ 1.  1.]

```

