## Zeros ##


**Scala:**
``` scala
val initMethod = Zeros

```
**Python:**
```python
init_method = Zeros()
```


Initialization method that set tensor to zeros.





**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val weightInitMethod = Zeros
val biasInitMethod = Zeros
val model = Linear(3, 2).setName("linear1")
model.setInitMethod(weightInitMethod, biasInitMethod)
println(model.getParametersTable().get("linear1").get)
```

```

 {
	weight: 0.0	0.0	0.0	
	        0.0	0.0	0.0	
	        [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
	bias: 0.0
	      0.0
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
weight_init = Zeros()
bias_init = Zeros()
model = Linear(3, 2)
model.set_init_method(weight_init, bias_init)
print("weight:")
print(model.get_weights()[0])
print("bias: ")
print(model.get_weights()[1])
```
```
creating: createZeros
creating: createZeros
creating: createLinear
weight:
[[ 0.  0.  0.]
 [ 0.  0.  0.]]
bias: 
[ 0.  0.]
```

