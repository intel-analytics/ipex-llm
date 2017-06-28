## RandomUniform ##


**Scala:**
``` scala
val initMethod = RandomUniform(lower: Double, upper: Double)

```
**Python:**
```python
init_method = RandomUniform(upper=None, lower=None, bigdl_type="float")
```

This initialization method draws samples from a uniform distribution. If the lower bound and upper bound of this uniform distribution is not specified, it will be set to [-limit, limit) where limit = 1/sqrt(fanIn).


**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val weightInitMethod = RandomUniform
val biasInitMethod = RandomUniform(0, 1)
val model = Linear(3, 2).setName("linear1")
model.setInitMethod(weightInitMethod, biasInitMethod)
println(model.getParametersTable().get("linear1").get)
```

```
 {
	weight: -0.572536	0.13046022	-0.040449623	
	        -0.547542	0.19093458	0.5632484	
	        [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
	bias: 0.785292
	      0.63280666
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
weight_init = RandomUniform()
bias_init = RandomUniform()
model = Linear(3, 2)
model.set_init_method(weight_init, bias_init)
print("weight:")
print(model.get_weights()[0])
print("bias: ")
print(model.get_weights()[1])
```
```
creating: createRandomUniform
creating: createRandomUniform
creating: createLinear
weight:
[[ 0.53153235  0.53016287  0.32831791]
 [-0.45736417 -0.16206641  0.21758588]]
bias: 
[ 0.32058391  0.26307678]

```

