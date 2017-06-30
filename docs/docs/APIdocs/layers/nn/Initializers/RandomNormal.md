## RandomNormal ##


**Scala:**
``` scala
val initMethod = RandomNormal(mean, stdv)

```
**Python:**
```python
init_method = RandomNormal(mean, stdv)
```
This initialization method draws samples from a normal distribution.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val weightInitMethod = RandomNormal(0, 1)
val biasInitMethod = RandomNormal(0, 1)
val linear = Linear(3, 2).setName("linear1")
linear.setInitMethod(weightInitMethod, biasInitMethod)
println(linear.getParametersTable().get("linear1").get)
```

```
 {
 	weight: -0.5908564	0.32844943	-0.845019	
 	        0.21550806	1.2037253	0.6807024	
 	        [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
 	bias: 0.5345903
 	      -0.76420456
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
from bigdl.nn.layer import *

weight_init = RandomNormal(0, 1)
bias_init = RandomNormal(0, 1)
linear= Linear(3, 2)
linear.set_init_method(weight_init, bias_init)
print("weight:")
print(linear.get_weights()[0])
print("bias: ")
print(linear.get_weights()[1])
```
```
creating: createRandomNormal
creating: createRandomNormal
creating: createLinear
weight:
[[-0.00784962  0.77845585 -1.16250944]
 [ 0.03195094 -0.15211993  0.6254822 ]]
bias: 
[-0.37883148 -0.81106091]

```

