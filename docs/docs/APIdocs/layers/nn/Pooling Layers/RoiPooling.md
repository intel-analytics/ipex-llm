## RoiPooling ##

**Scala:**
```scala
val m =  RoiPooling[T](pooled_w, pooled_h, spatial_scale)
```
**Python:**
```python
m = RoiPooling(pooled_w, pooled_h, spatial_scale)
```

RoiPooling is a module that performs Region of Interest pooling. 

It uses max pooling to convert the features inside any valid region of interest into a small feature map with a fixed spatial extent of pooledH × pooledW (e.g., 7 × 7).

An RoI is a rectangular window into a conv feature map. Each RoI is defined by a four-tuple (x1, y1, x2, y2) that specifies its top-left corner (x1, y1) and its bottom-right corner (x2, y2).

RoI max pooling works by dividing the h × w RoI window into an pooledH × pooledW grid of sub-windows of approximate size h/H × w/W and then max-pooling the values in each sub-window into the corresponding output grid cell. Pooling is applied independently to each feature map channel

`forward` accepts a table containing 2 tensors as input, the first tensor is the input image, the second tensor is the ROI regions. The dimension of the second tensor should be (*,5) (5 are  `batch_num, x1, y1, x2, y2`).  

**Scala example:**
```scala
scala> val input_data = Tensor[Float](2,2,6,8).randn()
input_data: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-1.6992253      1.2015936       -0.05117503     -0.7397121      1.0630354    0.8390283        -0.6759788      -2.5447173
-0.4982841      0.52384084      -0.7446032      0.59601253      -2.1328027   0.42816094       -0.48427275     0.80693465
0.23736824      1.8208236       1.7907403       1.3036019       -1.3374745   0.78269446       -0.38596377     0.5789667
1.3709409       1.3713696       -0.40264294     -0.0923695      0.9504029    -0.04030278      0.9314868       1.4269241
0.5179269       0.674084        -0.36575577     0.640429        0.8741701    -0.9512585       -0.04153871     -1.1504338
-0.535036       0.10304179      0.9589957       -1.8172264      -1.345329    1.0682544        0.20081596      0.13805625

(1,2,.,.) =
0.033531614     0.48715463      -1.2964486      0.08162945      -1.7955337   0.40367353       1.3779162       0.8270511
1.2370467       1.4615952       -0.7862223      -0.09450549     0.7965238    0.5320037        0.1332852       -0.28071782
-0.54606277     -1.12...

scala> import com.intel.analytics.bigdl.tensor.Storage
import com.intel.analytics.bigdl.tensor.Storage

scala> val rois = Array(0, 0, 0, 7, 5, 1, 6, 2, 7, 5, 1, 3, 1, 6, 4, 0, 3, 3, 3,   3)
rois: Array[Int] = Array(0, 0, 0, 7, 5, 1, 6, 2, 7, 5, 1, 3, 1, 6, 4, 0, 3, 3, 3, 3)

scala> val input_rois = Tensor(Storage(rois.map(x => x.toFloat))).resize(4, 5)
input_rois: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.0     0.0     0.0     7.0     5.0
1.0     6.0     2.0     7.0     5.0
1.0     3.0     1.0     6.0     4.0
0.0     3.0     3.0     3.0     3.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 4x5]

scala> val m = RoiPooling[Float](3, 2, 1)
m: com.intel.analytics.bigdl.nn.RoiPooling[Float] = nn.RoiPooling

scala> m.forward(T(input_data,input_rois))
res16: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
1.8208236       1.7907403       0.8390283
1.3713696       1.0682544       1.4269241

(1,2,.,.) =
1.4615952       0.7965238       1.901946
1.4465429       1.4202017       1.3160387

(2,1,.,.) =
-0.42326528     0.809365        0.809365
0.42272198      1.4816588       1.4816588

(2,2,.,.) =
1.5600494       1.5600494       0.93881196
0.7700858       1.4434724       1.4434724

(3,1,.,.) =
1.0095203       1.1964589       1.1964589
1.1247963       1.1247963       0.42272198

(3,2,.,.) =
1.0367005       1.0367005       0.3790048
0.31354976      0.31354976      1.5600494

(4,1,.,.) =
-0.0923695      -0.0923695      -0.0923695
-0.0923695      -0.0923695      -0.0923695

(4,2,.,.) =
1.4202017       1.4202017       1.4202017
1.4202017       1.4202017       1.4202017

[com.intel.analytics.bigdl.tensor.DenseTensor of size 4x2x2x3]

scala> val gradOut = Tensor[Float](4,2,2,3).randn()
gradOut: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-1.3592719      -1.2597874      0.47787797
-2.2111888      0.4890611       -0.050673995

(1,2,.,.) =
-0.70890087     0.8303274       -1.0301261
-0.26890132     -1.0551143      0.9207684

(2,1,.,.) =
-0.3968763      -0.5377336      0.55179757
0.2507654       0.5491315       -0.19382606

(2,2,.,.) =
0.04723405      -0.023283517    -0.276816
-0.35121053     0.39266664      0.08876784

(3,1,.,.) =
-1.6329603      1.1510664       -0.6817521
-0.36818376     1.9221923       -0.41676062

(3,2,.,.) =
-0.35098937     -0.94909745     1.1438006
0.6785622       -1.2131604      -0.031314004

(4,1,.,.) =
-0.30345625     -1.1790648      -0.9245161
-1.3502598      1.502442        -0.078317806

(4,2,.,.) =
0.9054519       0.70181507      -0.62081844
-0.5450406      -2.266202       -0.6190975

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 4x2x2x3]

scala> m.backward(T(input_data,input_rois),gradOut)
res17: com.intel.analytics.bigdl.utils.Table =
 {
        1: (1,1,.,.) =
           0.0  0.0     0.0     0.0     0.0     0.47787797      0.0     0.0
           0.0  0.0     0.0     0.0     0.0     0.0     0.0     0.0
           0.0  -1.3592719      -1.2597874      0.0     0.0     0.0     0.0  0.0
           0.0  -2.2111888      0.0     -2.333173       0.0     0.0     0.0  -0.050673995
           0.0  0.0     0.0     0.0     0.0     0.0     0.0     0.0
           0.0  0.0     0.0     0.0     0.0     0.4890611       0.0     0.0

           (1,2,.,.) =
           0.0  0.0     0.0     0.0     0.0     0.0     0.0     0.0
           0.0  -0.70890087     0.0     0.0     0.8303274       0.0     0.0  0.0
           0.0  0.0     0.0     0.0     0.0     0.0     0.0     -1.0301261
           0.0  0.0     0.0     -3.4990058      0.0     0.9207684       0.0  0.0
           0.0  -0.26890132     0.0     0.0     0.0     0.0     0.0     0.0
           0.0  0.0     0.0     0.0     0.0     0.0     0.0     0.0

           (2,1,.,.) =
           0.0  0.0     0.0     0.0     0.0     0.0     0.0     0.0
           0.0  0.0     0.0     0.0     -1.6329603      0.0     0.0     0.0
           0.0  0.0     0.0     0.0     0.0     0.46931434      -0.3968763   0.0
           0.0  0.0     0.0     0.0     1...
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input_data = np.random.rand(2,2,6,8)
input_rois = np.array([0, 0, 0, 7, 5, 1, 6, 2, 7, 5, 1, 3, 1, 6, 4, 0, 3, 3, 3, 3],dtype='float64').reshape(4,5)
#print "input is :",input_data, input_rois

m = RoiPooling(3,2,1,bigdl_type='float')
out = m.forward([input_data,input_rois])
print "output of m is :",out
```
