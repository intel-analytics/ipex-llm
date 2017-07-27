You can specify the initialization strategy via `setInitMethod`, but it only support `Linear`, `RNN`, `SpatialConvolution`, `SpatialDilatedConvolution` and `SpatialFullConvolution` for now. For those who doesn't provide `setInitMethod` would have their own `Default` initialization strategy. Most of them are Uniform, except that LookupTable use normal(0, 1) and PReLU use constant 0.25.

## xavier ##


**scala:**
``` scala
val initmethod = xavier

```
**python:**
```python
init_method = "xavier"
```


the xavier initialization method draws samples from a uniform distribution
bounded by [-limit, limit) where limit = sqrt(6.0/(fanin+fanout)). the rationale
behind this formula can be found in the paper
[understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).



**scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.tensornumericmath.tensornumeric.numericfloat

val weightinitmethod = xavier
val biasinitmethod = xavier
val model = linear(3, 2).setname("linear1")
model.setinitmethod(weightinitmethod, biasinitmethod)
println(model.getparameterstable().get("linear1").get)
```

```
 {
	weight: -0.78095555	-0.09939616	0.12034761	
	        -0.3019594	0.11734331	0.80369484	
	        [com.intel.analytics.bigdl.tensor.densetensor of size 2x3]
	bias: 1.0727772
	      -0.6703765
	      [com.intel.analytics.bigdl.tensor.densetensor of size 2]
	gradbias: 0.0
	          0.0
	          [com.intel.analytics.bigdl.tensor.densetensor of size 2]
	gradweight: 0.0	0.0	0.0	
	            0.0	0.0	0.0	
	            [com.intel.analytics.bigdl.tensor.densetensor of size 2x3]
 }


```

**python example:**
```python
from bigdl.nn.initialization_method import *
weight_init = "xavier"
bias_init = "xavier"
model = linear(3, 2)
model.set_init_method(weight_init, bias_init)
print("weight:")
print(model.get_weights()[0])
print("bias: ")
print(model.get_weights()[1])
```
```
creating: createxavier
creating: createxavier
creating: createlinear
weight:
[[ 0.00580597 -0.73662472  0.13767919]
 [ 0.16802482 -0.49394709 -0.74967551]]
bias: 
[-1.12355328  0.0779365 ]
```


## bilinearfiller ##


**scala:**
``` scala
val initmethod = bilinearfiller

```
**python:**
```python
init_method = bilinearfiller()
```

initialize the weight with coefficients for bilinear interpolation. a common use case is with the deconvolutionlayer acting as upsampling. this initialization method can only be used in the weight initialization of spatialfullconvolution.


**scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.tensornumericmath.tensornumeric.numericfloat

val weightinitmethod = bilinearfiller
val biasinitmethod - zeros
val model = spatialfullconvolution(2, 3, 2, 2).setname("sfconv")
model.setinitmethod(weightinitmethod, biasinitmethod)
println(model.getparameterstable().get("sfconv").get)
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
	        
	        [com.intel.analytics.bigdl.tensor.densetensor of size 1x2x3x2x2]
	bias: 0.0
	      0.0
	      0.0
	      [com.intel.analytics.bigdl.tensor.densetensor of size 3]
	gradbias: 0.0
	          0.0
	          0.0
	          [com.intel.analytics.bigdl.tensor.densetensor of size 3]
	gradweight: (1,1,1,.,.) =
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
	            
	            [com.intel.analytics.bigdl.tensor.densetensor of size 1x2x3x2x2]
 }

```

**python example:**
```python
from bigdl.nn.initialization_method import *
weight_init = bilinearfiller()
bias_init = zeros()
model =  spatialfullconvolution(2, 3, 2, 2)
model.set_init_method(weight_init, bias_init)
print("weight:")
print(model.get_weights()[0])
print("bias: ")
print(model.get_weights()[1])
```
```
creating: createbilinearfiller
creating: createzeros
creating: createspatialfullconvolution
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
