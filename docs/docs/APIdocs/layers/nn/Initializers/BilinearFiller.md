## BilinearFiller ##


**Scala:**
``` scala
val initMethod = BilinearFiller

```
**Python:**
```python
init_method = BilinearFiller(bigdl_type="float")
```

Initialize the weight with coefficients for bilinear interpolation. A common use case is with the DeconvolutionLayer acting as upsampling. This initialization method can only be used in the weight initialization of SpatialFullConvolution.


**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val weightInitMethod = BilinearFiller
val model = SpatialFullConvolution(2, 3, 2, 2).setName("sfconv")
model.setInitMethod(weightInitMethod, Zeros)
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

