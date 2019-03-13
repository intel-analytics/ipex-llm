

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

---
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

## ConstInitMethod ##


**Scala:**
``` scala
val initMethod = ConstInitMethod(value: Double)

```
**Python:**
```python
init_method = ConstInitMethod(value)
```

Initialization method that set tensor to the specified constant value.


**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat


val weightInitMethod = ConstInitMethod(0.2)
val biasInitMethod = ConstInitMethod(0.2)
val linear = Linear(3, 2).setName("linear1")
linear.setInitMethod(weightInitMethod, biasInitMethod)
println(linear.getParametersTable().get("linear1").get)
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
linear = Linear(3, 2)
linear.set_init_method(weight_init, bias_init)
print("weight:")
print(linear.get_weights()[0])
print("bias: ")
print(linear.get_weights()[1])
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


## BilinearFiller ##


**Scala:**
``` scala
val initMethod = BilinearFiller

```
**Python:**
```python
init_method = BilinearFiller()
```

Initialize the weight with coefficients for bilinear interpolation. A common use case is with the DeconvolutionLayer acting as upsampling. This initialization method can only be used in the weight initialization of SpatialFullConvolution.


**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val weightInitMethod = BilinearFiller
val biasInitMethod = Zeros
val model = SpatialFullConvolution(2, 3, 2, 2).setName("sfconv")
model.setInitMethod(weightInitMethod, biasInitMethod)
println(model.getParametersTable().get("sfconv").get)
```

```
{
	weight: (1,1,1,.,.) =
	        1.0	0.0	
	        0.0	0.0	
	        
	        (1,1,2,.,.) =
	        1.0	0.0	
	        0.0	0.0	
	        
	        (1,1,3,.,.) =
	        1.0	0.0	
	        0.0	0.0	
	        
	        (1,2,1,.,.) =
	        1.0	0.0	
	        0.0	0.0	
	        
	        (1,2,2,.,.) =
	        1.0	0.0	
	        0.0	0.0	
	        
	        (1,2,3,.,.) =
	        1.0	0.0	
	        0.0	0.0	
	        
	        [com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x3x2x2]
	bias: 0.0
	      0.0
	      0.0
	      [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
	gradBias: 0.0
	          0.0
	          0.0
	          [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
	gradWeight: (1,1,1,.,.) =
	            0.0	0.0	
	            0.0	0.0	
	            
	            (1,1,2,.,.) =
	            0.0	0.0	
	            0.0	0.0	
	            
	            (1,1,3,.,.) =
	            0.0	0.0	
	            0.0	0.0	
	            
	            (1,2,1,.,.) =
	            0.0	0.0	
	            0.0	0.0	
	            
	            (1,2,2,.,.) =
	            0.0	0.0	
	            0.0	0.0	
	            
	            (1,2,3,.,.) =
	            0.0	0.0	
	            0.0	0.0	
	            
	            [com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x3x2x2]
 }

```

**Python example:**
```python
from bigdl.nn.initialization_method import *
weight_init = BilinearFiller()
bias_init = Zeros()
model =  SpatialFullConvolution(2, 3, 2, 2)
model.set_init_method(weight_init, bias_init)
print("weight:")
print(model.get_weights()[0])
print("bias: ")
print(model.get_weights()[1])
```
```
creating: createBilinearFiller
creating: createZeros
creating: createSpatialFullConvolution
weight:
[[[[[ 1.  0.]
    [ 0.  0.]]

   [[ 1.  0.]
    [ 0.  0.]]

   [[ 1.  0.]
    [ 0.  0.]]]


  [[[ 1.  0.]
    [ 0.  0.]]

   [[ 1.  0.]
    [ 0.  0.]]

   [[ 1.  0.]
    [ 0.  0.]]]]]
bias: 
[ 0.  0.  0.]


```


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




## RandomUniform ##


**Scala:**
``` scala
val initMethod = RandomUniform(lower, upper)

```
**Python:**
```python
init_method = RandomUniform(upper=None, lower=None)
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


## Define your own Initializer ##

All customizedInitializer should implement the `InitializationMethod` trait

```scala
/**
 * Initialization method to initialize bias and weight.
 * The init method will be called in Module.reset()
 */

trait InitializationMethod {

  type Shape = Array[Int]

  /**
   * Initialize the given variable
   *
   * @param variable    the variable to initialize
   * @param dataFormat  describe the meaning of each dimension of the variable
   */
  def init[T](variable: Tensor[T], dataFormat: VariableFormat)
             (implicit ev: TensorNumeric[T]): Unit
}
```
The [RandomUniform](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/nn/InitializationMethod.scala#L163)
code should give you a good sense of how to implement this trait.


_**Python**
Custom initialization method in python is not supported right now.
