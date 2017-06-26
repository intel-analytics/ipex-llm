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
val spatialSubtractiveNormalization = SpatialSubtractiveNormalization(4, kernel)

> print(spatialSubtractiveNormalization.forward(Tensor(3, 4, 5, 5).rand()))
(1,1,.,.) =
-0.23341611	0.3344916	0.23259735	-0.30317286	0.0602113	
0.2494207	-0.09478736	0.43968314	0.015044361	0.08818066	
0.19935995	0.22641534	-0.16797271	0.028918773	0.02103293	
0.34971905	-0.36060834	0.35384625	0.37494865	-0.21338695	
0.3034316	0.18821329	0.24799132	-0.47940618	0.102709115	

(1,2,.,.) =
-0.39597827	-0.14664972	0.17417371	-0.17144218	0.42531347	
0.17727757	-0.064986885	0.32195455	-0.30659157	0.28571957	
0.18861032	-0.24977896	0.16757923	-0.20946856	0.22372705	
0.35106254	0.14381915	-0.329951	0.48048618	0.44545975	
0.24326172	0.08867538	-0.46260777	-0.24209261	0.0938741	
.......

```

**Python example:**
```python
from bigdl.nn.layer import *
kernel=np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
spatialSubtractiveNormalization = SpatialSubtractiveNormalization(1, kernel)
>  spatialSubtractiveNormalization.forward(np.array([[[[1, 2, 3, 4, 5]]]]))
[array([[[[ 0.,  0.,  0.,  0.,  0.]]]], dtype=float32)]

     
```

