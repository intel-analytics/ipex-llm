## SpatialZeroPadding ##

**Scala:**
```scala
val spatialZeroPadding = SpatialZeroPadding(padLeft: Int, padRight: Int, padTop: Int, padBottom: Int)
```
**Python:**
```python
spatialZeroPadding = SpatialZeroPadding(pad_left, pad_right, pad_top, pad_bottom)
```

Each feature map of a given input is padded with specified number of zeros.
 
If padding values are negative, then input will be cropped.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val spatialZeroPadding = SpatialZeroPadding(1, 0, -1, 0)
val input = Tensor(3, 3, 3).rand()
> print(input)
(1,.,.) =
0.9494078	0.31556255	0.8432871	
0.0064580487	0.6487367	0.151881	
0.8822722	0.3634125	0.7034494	

(2,.,.) =
0.32691675	0.07487922	0.08813124	
0.4564806	0.37191486	0.05507739	
0.10097649	0.6589037	0.8721945	

(3,.,.) =
0.068939745	0.040364727	0.4893642	
0.39481318	0.17923461	0.15748173	
0.87117475	0.9933199	0.6097995

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3x3]

>  print(spatialZeroPadding.forward(input))
(1,.,.) =
0.0	0.0064580487	0.6487367	0.151881	
0.0	0.8822722	0.3634125	0.7034494	

(2,.,.) =
0.0	0.4564806	0.37191486	0.05507739	
0.0	0.10097649	0.6589037	0.8721945	

(3,.,.) =
0.0	0.39481318	0.17923461	0.15748173	
0.0	0.87117475	0.9933199	0.6097995	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2x4]

```

**Python example:**
```python
from bigdl.nn.layer import *
spatialZeroPadding = SpatialZeroPadding(1, 0, -1, 0)
> spatialZeroPadding.forward(np.array([[[1, 2],[3, 4]],[[1, 2],[3, 4]]]))
[array([[[ 0.,  3.,  4.]],
       [[ 0.,  3.,  4.]]], dtype=float32)]
       
```


