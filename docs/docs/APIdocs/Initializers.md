You can specify the initialization strategy via `init_method` within the constructor, but it only supported in `Linear`, `RNN`, `SpatialConvolution`, `SpatialDilatedConvolution` and `SpatialFullConvolution` for now. For those who doesn't provide `init_method` would have their own `Default` initialization strategy. 

## Default ##
Each layer would has it's own default strategy, but most of them are Uniform, except that LookupTable use normal(0, 1) and PReLU use constant 0.25.

**scala example:**
```scala
      import com.intel.analytics.bigdl.nn._
      import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
      val model = Linear(3, 2, initMethod = Default)
      println(model.weight)
      println(model.bias)
```

```
-0.15182142	-0.52394074	-0.39509705	
0.5540198	0.02407885	-0.33133075	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
0.06908254
0.3079934
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
```

**python example:**
```python
from bigdl.nn.layer import *
model = Linear(3, 2, init_method="Default").set_name("linear1")
print(model.parameters()["linear1"]["weight"])
print(model.parameters()["linear1"]["bias"])
```
```
[[-0.55911756  0.57299054  0.27762395]
 [-0.14510377  0.41430488  0.38458899]]
[-0.2382445  -0.45159951]

```


## xavier ##

The xavier initialization method draws samples from a uniform distribution
bounded by [-limit, limit) where limit = sqrt(6.0/(fanin+fanout)). the rationale
behind this formula can be found in the paper
[understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).



**scala example:**
```scala
      import com.intel.analytics.bigdl.nn._
      import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
      val model = Linear(3, 2, initMethod = Xavier)
      println(model.weight)
      println(model.bias)
```

```
-0.36605954	-0.1555609	0.28505832	
0.87636554	0.83877754	-0.76975846	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
0.0
0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2] 


```

**python example:**
```python
from bigdl.nn.layer import *
model = Linear(3, 2, init_method="Xavier").set_name("linear1")
print(model.parameters()["linear1"]["weight"])
print(model.parameters()["linear1"]["bias"])
```
```
[[-0.90043104  0.27011687  0.98132485]
 [-0.0541487   0.15733482 -0.41851035]]
[ 0.  0.]
```


## bilinearfiller ##


Initialize the weight with coefficients for bilinear interpolation. a common use case is with the deconvolutionlayer acting as upsampling. this initialization method can only be used in the weight initialization of spatialfullconvolution.


**scala example:**
```scala
      import com.intel.analytics.bigdl.nn._
      import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
      val model = SpatialFullConvolution(1, 1, 3, 3, initMethod = BilinearFiller)
      println(model.weight)
      println(model.bias)
```

```
(1,1,1,.,.) =
0.0625	0.1875	0.1875	
0.1875	0.5625	0.5625	
0.1875	0.5625	0.5625	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x1x3x3]
0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1]
```

**python example:**
```python
from bigdl.nn.layer import *
model = SpatialFullConvolution(1, 1, 3, 3, init_method="BilinearFiller").set_name("sfc1")
print(model.parameters()["sfc1"]["weight"])
print(model.parameters()["sfc1"]["bias"])

```

```
[[[[[ 0.0625  0.1875  0.1875]
    [ 0.1875  0.5625  0.5625]
    [ 0.1875  0.5625  0.5625]]]]]
[ 0.]

```
