## Xavier ##


**Scala:**
``` scala
val initMethod = Xavier

```
**Python:**
```python
init_method = Xavier()
```


The Xavier initialization method draws samples from a uniform distribution
bounded by [-limit, limit) where limit = sqrt(6.0/(fanIn+fanOut)). The rationale
behind this formula can be found in the paper
[Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).



**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val weightInitMethod = Xavier
val biasInitMethod = Xavier
val model = Linear(3, 2).setName("linear1")
model.setInitMethod(weightInitMethod, biasInitMethod)
println(model.getParametersTable().get("linear1").get)
```

```
 {
	weight: -0.78095555	-0.09939616	0.12034761	
	        -0.3019594	0.11734331	0.80369484	
	        [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
	bias: 1.0727772
	      -0.6703765
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
weight_init = Xavier()
bias_init = Xavier()
model = Linear(3, 2)
model.set_init_method(weight_init, bias_init)
print("weight:")
print(model.get_weights()[0])
print("bias: ")
print(model.get_weights()[1])
```
```
creating: createXavier
creating: createXavier
creating: createLinear
weight:
[[ 0.00580597 -0.73662472  0.13767919]
 [ 0.16802482 -0.49394709 -0.74967551]]
bias: 
[-1.12355328  0.0779365 ]
```

