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

val input = Tensor(2, 2, 2, 2).rand()

> print(input)
(1,1,.,.) =
0.42596373	0.20075735	
0.10307904	0.7486494	

(1,2,.,.) =
0.9887414	0.3554662	
0.6291069	0.53952795	

(2,1,.,.) =
0.41220918	0.5463298	
0.40766734	0.08064394	

(2,2,.,.) =
0.58255607	0.027811589	
0.47811228	0.3082057	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x2x2]

> print(spatialCrossMapLRN.forward(input))
(1,1,.,.) =
0.42522463	0.20070718	
0.10301625	0.74769455	

(1,2,.,.) =
0.98702586	0.35537735	
0.6287237	0.5388398	

(2,1,.,.) =
0.41189456	0.5460847	
0.4074261	0.08063166	

(2,2,.,.) =
0.5821114	0.02779911	
0.47782937	0.3081588	

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

