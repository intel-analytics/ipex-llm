## SpatialSubtractiveNormalization ##

**Scala:**
```scala
val spatialSubtractiveNormalization = SpatialSubtractiveNormalization(val nInputPlane: Int = 1, var kernel: Tensor[T] = null)
```
**Python:**
```python
spatialSubtractiveNormalization = SpatialSubtractiveNormalization(n_input_plane=1, kernel=None)
```

SpatialSubtractiveNormalization applies a spatial subtraction operation on a series of 2D inputs using kernel for computing the weighted average in a neighborhood.The neighborhood is defined for a local spatial region that is the size as kernel and across all features. For an input image, since there is only one feature, the region is only spatial. For an RGB image, the weighted average is taken over RGB channels and a spatial region.

If the kernel is 1D, then it will be used for constructing and separable 2D kernel.
The operations will be much more efficient in this case.
 
The kernel is generally chosen as a gaussian when it is believed that the correlation
of two pixel locations decrease with increasing distance. On the feature dimension,
a uniform average is used since the weighting across features is not known.

```
nInputPlane : number of input plane, default is 1.
kernel : kernel tensor, default is a 9 x 9 tensor.
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val kernel = Tensor(3, 3).rand()

> print(kernel)
0.56141114	0.76815456	0.29409808	
0.3599753	0.17142025	0.5243272	
0.62450963	0.28084084	0.17154165	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]


val spatialSubtractiveNormalization = SpatialSubtractiveNormalization(1, kernel)

val input = Tensor(1, 1, 1, 5).rand()

> print(input)
(1,1,.,.) =
0.122356184	0.44442436	0.6394927	0.9349956	0.8226007	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x1x1x5]

> print(spatialSubtractiveNormalization.forward(input))
(1,1,.,.) =
-0.2427161	0.012936085	-0.08024883	0.15658027	-0.07613802	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x1x5]


```

**Python example:**
```python
from bigdl.nn.layer import *
kernel=np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
spatialSubtractiveNormalization = SpatialSubtractiveNormalization(1, kernel)
>  spatialSubtractiveNormalization.forward(np.array([[[[1, 2, 3, 4, 5]]]]))
[array([[[[ 0.,  0.,  0.,  0.,  0.]]]], dtype=float32)]

     
```

