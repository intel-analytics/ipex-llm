## SpatialCrossMapLRN ##

**Scala:**
```scala
val spatialCrossMapLRN = SpatialCrossMapLRN(val size: Int = 5, val alpha: Double = 1.0, val beta: Double = 0.75, val k: Double = 1.0)
```
**Python:**
```python
spatialCrossMapLRN = SpatialCrossMapLRN(size=5,alpha=1.0,beta=0.75,k=1.0)
```

SpatialCrossMapLRN applies Spatial Local Response Normalization between different feature maps

```
                             x_f
  y_f =  -------------------------------------------------
          (k+(alpha/size)* sum_{l=l1 to l2} (x_l^2^))^beta^
          
where  l1 corresponds to `max(0,f-ceil(size/2))` and l2 to `min(F, f-ceil(size/2) + size)`, `F` is the number  of feature maps       
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val spatialCrossMapLRN = SpatialCrossMapLRN(5, 0.01, 0.75, 1.0)

> print(spatialCrossMapLRN.forward(Tensor(2, 2, 2, 2).rand()))
(1,1,.,.) =
0.36764133	0.39759055	
0.8221005	0.95572835	

(1,2,.,.) =
0.72927773	0.7172886	
0.7174055	0.5311008	

(2,1,.,.) =
0.7350153	0.599728	
0.04587644	0.85535294	

(2,2,.,.) =
0.7313947	0.5809075	
0.19847111	0.3734013	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2]


```

**Python example:**
```python
from bigdl.nn.layer import *
spatialCrossMapLRN = SpatialCrossMapLRN(5, 0.01, 0.75, 1.0)
> spatialCrossMapLRN.forward(np.array([[[[1, 2],[3, 4]],[[5, 6],[7, 8]]],[[[9, 10],[11, 12]],[[13, 14],[15, 16]]]]))
[array([[[[  0.96269381,   1.88782692],
         [  2.76295042,   3.57862759]],

        [[  4.81346893,   5.66348076],
         [  6.44688463,   7.15725517]]],


       [[[  6.6400919 ,   7.05574226],
         [  7.41468   ,   7.72194815]],

        [[  9.59124374,   9.87803936],
         [ 10.11092758,  10.29593086]]]], dtype=float32)]

     
```

