## **Dropout**
Applies Dropout to the input by randomly setting a fraction 'p' of input units to 0 at each update during training time in order to prevent overfitting.

**Scala:**
```scala
Dropout(p, inputShape = null)
```
**Python:**
```python
Dropout(p, input_shape=None, name=None)
```

**Parameters:**

* `p`: Fraction of the input units to drop. Between 0 and 1.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.Dropout
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Dropout[Float](0.3, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.5496527       0.34846303      1.8184849       -0.8750735
-0.2907603      0.124056354     -0.5447822      -0.34512782
1.003834        -0.27847317     -0.16524693     -0.12172801

(2,.,.) =
-0.50297844     -0.78188837     -1.5617784      -1.2353797
-1.5052266      -1.6246556      0.5203618       1.144502
-0.18044183     -0.032648038    -1.9599762      -0.6970337

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
2.2137897       0.49780434      2.5978355       -1.250105
0.0     0.17722337      0.0     0.0
1.4340487       0.0     0.0     -0.17389716

(2,.,.) =
-0.71854067     -1.1169834      -2.231112       -1.7648282
-2.1503239      -2.3209367      0.743374        1.635003
-0.25777406     0.0     -2.799966       -0.99576247

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Dropout
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Dropout(0.3, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
array([[[0.80667346, 0.5370812 , 0.59039134, 0.48676815],
        [0.70987046, 0.11246564, 0.68062359, 0.48074257],
        [0.61979472, 0.36682032, 0.08320745, 0.41117697]],

       [[0.19616717, 0.18093539, 0.52080897, 0.73326568],
        [0.72752776, 0.81963229, 0.05652756, 0.37253947],
        [0.70200807, 0.27836313, 0.24421078, 0.58191582]]])
```
Output is
```python
array([[[1.1523907 , 0.7672588 , 0.        , 0.6953831 ],
        [1.0141007 , 0.1606652 , 0.9723194 , 0.6867751 ],
        [0.        , 0.5240291 , 0.11886779, 0.58739567]],

       [[0.2802388 , 0.        , 0.74401283, 1.0475224 ],
        [1.0393254 , 1.1709033 , 0.08075366, 0.53219926],
        [1.0028687 , 0.39766163, 0.        , 0.8313083 ]]], dtype=float32)
```

---
## **SpatialDropout2D**
Spatial 2D version of Dropout.

This version performs the same function as Dropout, however it drops entire 2D feature maps instead of individual elements. If adjacent pixels within feature maps are strongly correlated (as is normally the case in early convolution layers) then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, SpatialDropout2D will help promote independence between feature maps and should be used instead.

The input of this layer should be 4D tensor with shape: (samples, channels, rows, cols) if data_format='th' (Channel First) or 4D tensor with shape: (samples, rows, cols, channels) if data_format='tf' (Channel Last).

**Scala:**
```scala
SpatialDropout2D(p = 0.5, dimOrdering = "th", inputShape = null)
```
**Python:**
```python
SpatialDropout2D(p=0.5, dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `p`: Fraction of the input units to drop. Between 0 and 1.
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.SpatialDropout2D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(SpatialDropout2D[Float](inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
1.266674        -0.19261484     0.8210725       -0.22291088
-0.38138267     1.7019615       1.1729054       0.59097356
-0.50952524     -1.9868233      -0.17180282     -1.2743127

(1,2,.,.) =
-0.13727586     -0.7740464      1.2427979       -0.46285817
-1.747042       1.3353567       1.1310997       -0.26019064
0.9580778       -0.69689065     -0.77704996     0.704949

(2,1,.,.) =
0.040080033     0.08806901      0.44471294      0.4693497
-1.2577269      -2.5343444      -0.5290871      0.73988694
-0.4042877      -0.20460072     -0.68553877     0.59006995

(2,2,.,.) =
-0.06227895     -0.9075216      1.226318        1.0563084
-0.6985987      -0.20155957     0.1005844       -0.49736363
1.3935218       -2.8411357      -1.6742039      0.26154035

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.0     -0.0    0.0     -0.0
-0.0    0.0     0.0     0.0
-0.0    -0.0    -0.0    -0.0

(1,2,.,.) =
-0.13727586     -0.7740464      1.2427979       -0.46285817
-1.747042       1.3353567       1.1310997       -0.26019064
0.9580778       -0.69689065     -0.77704996     0.704949

(2,1,.,.) =
0.040080033     0.08806901      0.44471294      0.4693497
-1.2577269      -2.5343444      -0.5290871      0.73988694
-0.4042877      -0.20460072     -0.68553877     0.59006995

(2,2,.,.) =
-0.06227895     -0.9075216      1.226318        1.0563084
-0.6985987      -0.20155957     0.1005844       -0.49736363
1.3935218       -2.8411357      -1.6742039      0.26154035

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import SpatialDropout2D

model = Sequential()
model.add(SpatialDropout2D(input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.45638721 0.87479404 0.28319946 0.85046252]
   [0.90687581 0.29446766 0.23341603 0.92425726]
   [0.51232495 0.83895807 0.90536451 0.41231943]]

  [[0.00397271 0.28512243 0.32912336 0.27304027]
   [0.97274043 0.92907157 0.25843125 0.201849  ]
   [0.42783297 0.91400856 0.19290376 0.83749261]]]


 [[[0.03282751 0.60866148 0.47616452 0.4300911 ]
   [0.75731354 0.34609462 0.66514783 0.18193801]
   [0.6748754  0.94068849 0.38504096 0.66447561]]

  [[0.61274329 0.56573389 0.21795374 0.45314279]
   [0.2883045  0.22641016 0.83014439 0.21362862]
   [0.33618578 0.47346473 0.96971251 0.2937416 ]]]]
```
Output is
```python
[[[[0.45638722 0.87479407 0.28319946 0.8504625 ]
   [0.9068758  0.29446766 0.23341602 0.9242573 ]
   [0.5123249  0.8389581  0.9053645  0.41231942]]

  [[0.00397271 0.28512242 0.32912338 0.27304026]
   [0.9727404  0.92907155 0.25843126 0.201849  ]
   [0.42783296 0.91400856 0.19290376 0.8374926 ]]]


 [[[0.03282751 0.6086615  0.47616452 0.4300911 ]
   [0.75731355 0.3460946  0.66514784 0.18193801]
   [0.6748754  0.9406885  0.38504097 0.6644756 ]]

  [[0.         0.         0.         0.        ]
   [0.         0.         0.         0.        ]
   [0.         0.         0.         0.        ]]]]
```
