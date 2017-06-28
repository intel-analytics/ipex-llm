## ConstInitMethod ##


**Scala:**
``` scala
val initMethod = ConstInitMethod(value: Double)

```
**Python:**
```python
init_method = ConstInitMethod(value, bigdl_type="float")
```

Initialization method that set tensor to the specified constant value.


**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat


val weightInitMethod = ConstInitMethod(0.2)
val biasInitMethod = ConstInitMethod(0.2)
val model = Linear(3, 2).setName("linear1")
model.setInitMethod(weightInitMethod, biasInitMethod)
println(model.getParametersTable().get("linear1").get)
```

```
 {
	weight: 0.2	0.2	0.2
	        0.2	0.2	0.2
	        [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
	bias: 0.2
	      0.2
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
weight_init = ConstInitMethod(0.2)
bias_init = ConstInitMethod(0.2)
model = Linear(3, 2)
model.set_init_method(weight_init, bias_init)
print("weight:")
print(model.get_weights()[0])
print("bias: ")
print(model.get_weights()[1])
```
```
creating: createConstInitMethod
creating: createConstInitMethod
creating: createLinear
weight:
[[ 0.2  0.2  0.2]
 [ 0.2  0.2  0.2]]
bias:
[ 0.2  0.2]

```

