## **SpatialDropout3D**
Spatial 3D version of Dropout.

This version performs the same functionalities as Dropout, however it drops entire 3D feature maps instead of individual elements. If adjacent voxels within feature maps are strongly correlated (as is normally the case in early convolution layers) then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, SpatialDropout3D will help promote independence between feature maps and should be used instead.

The input is 5D tensor with shape: (batch_size, channels, dim1, dim2, dim3)

**Scala:**
```scala
SpatialDropout3D(p = 0.5, dimOrdering = "th", inputShape = null)
```
**Python:**
```python
SpatialDropout3D(p=0.5, dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `p`: Fraction of the input units to drop. Between 0 and 1.
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.SpatialDropout3D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(SpatialDropout3D[Float](inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
1.5842006       -1.486708       -1.0261744
-0.8227147      0.1386223       -0.46191332

(1,1,2,.,.) =
-0.7794714      0.52259976      1.5326598
0.32597166      0.84018683      -0.24034925

(1,2,1,.,.) =
0.5037644       -0.42065156     1.1590574
1.4855213       -1.4098096      0.5154563

(1,2,2,.,.) =
2.1119535       0.4159602       -0.33109334
-1.9544226      0.014503485     -0.7715549

(2,1,1,.,.) =
1.1496683       0.20273614      -2.6363356
-1.6820912      -1.1656585      -0.8387814

(2,1,2,.,.) =
-1.1125584      -1.9073812      0.78532314
-1.0033096      -0.24038585     1.0534006

(2,2,1,.,.) =
0.46944886      -1.8767697      0.7275591
0.36211884      0.34403932      -1.3721423

(2,2,2,.,.) =
0.37117565      -0.45195773     0.66517854
0.3873176       -1.8218406      1.9105781

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
0.0     -0.0    -0.0
-0.0    0.0     -0.0

(1,1,2,.,.) =
-0.0    0.0     0.0
0.0     0.0     -0.0

(1,2,1,.,.) =
0.0     -0.0    0.0
0.0     -0.0    0.0

(1,2,2,.,.) =
0.0     0.0     -0.0
-0.0    0.0     -0.0

(2,1,1,.,.) =
0.0     0.0     -0.0
-0.0    -0.0    -0.0

(2,1,2,.,.) =
-0.0    -0.0    0.0
-0.0    -0.0    0.0

(2,2,1,.,.) =
0.0     -0.0    0.0
0.0     0.0     -0.0

(2,2,2,.,.) =
0.0     -0.0    0.0
0.0     -0.0    0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import SpatialDropout3D

model = Sequential()
model.add(SpatialDropout3D(input_shape=(2, 2, 2, 2)))
input = np.random.random([2, 2, 2, 2, 2])
output = model.forward(input)
```
Input is:
```python
[[[[[0.19861794 0.32822715]
    [0.78735804 0.0586697 ]]

   [[0.22181565 0.09894792]
    [0.43668179 0.22321872]]]


  [[[0.81122679 0.44084158]
    [0.70199098 0.10383273]]

   [[0.78102397 0.62514588]
    [0.6933126  0.7830806 ]]]]



 [[[[0.22229716 0.90939922]
    [0.2453606  0.49500498]]

   [[0.95518136 0.78983711]
    [0.724247   0.62801332]]]


  [[[0.89800761 0.5523274 ]
    [0.83153558 0.58200981]]

   [[0.84787731 0.16651971]
    [0.22528241 0.68706778]]]]]
```
Output is
```python
[[[[[0.19861795 0.32822713]
    [0.78735805 0.0586697 ]]

   [[0.22181565 0.09894791]
    [0.43668178 0.22321871]]]


  [[[0.8112268  0.4408416 ]
    [0.70199096 0.10383273]]

   [[0.781024   0.62514585]
    [0.6933126  0.7830806 ]]]]



 [[[[0.         0.        ]
    [0.         0.        ]]

   [[0.         0.        ]
    [0.         0.        ]]]


  [[[0.89800763 0.5523274 ]
    [0.8315356  0.5820098 ]]

   [[0.8478773  0.16651972]
    [0.22528242 0.6870678 ]]]]]
```

---
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
## **GaussianDropout**
Apply multiplicative 1-centered Gaussian noise.

As it is a regularization layer, it is only active at training time.

**Scala:**
```scala
GaussianDropout(p, inputShape = null)
```
**Python:**
```python
GaussianDropout(p, input_shape=None, name=None)
```

**Parameters:**

* `p`: Drop probability (as with 'Dropout'). The multiplicative noise will have standard deviation 'sqrt(p/(1-p))'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.
* `name`: String to set the name of the layer. If not specified, its name will by default to be a generated string.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.GaussianDropout
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GaussianDropout[Float](0.45, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.8969221   2.454179    -0.26737544 0.86235714
-0.61781764 -0.48739514 0.2337097   1.0086832
1.7666794   -1.120229   -0.28245732 0.845279

(2,.,.) =
1.2763704   -0.3854067  0.0061038486    0.931373
0.67848265  -3.098805   -0.1240183  0.36834922
0.9772534   -0.639048   -0.078967154    1.4179249

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
1.8251847   4.3458977   -0.6353459  -0.10734326
-0.4009521  -0.5479114  0.1226105   2.0534828
-0.03313    -2.271632   0.122886114 -0.44396263

(2,.,.) =
0.45101312  -0.48233575 0.008046541 2.2945886
1.3415622   -1.9070724  -0.1681036  0.60575134
0.88338673  -1.4186113  -0.012104415    0.3102114

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import GaussianDropout
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(GaussianDropout(0.45, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
array([[[ 0.55167758,  0.07427833,  0.59777983,  0.86986969],
        [ 0.53097779,  0.4174687 ,  0.58065922,  0.73479602],
        [ 0.43731939,  0.64465237,  0.32946076,  0.59878638]],

       [[ 0.26428987,  0.29575131,  0.36229906,  0.66938424],
        [ 0.74325536,  0.08672916,  0.35460851,  0.00122828],
        [ 0.27095285,  0.09442922,  0.02280022,  0.68735133]]])
```
Output is
```python
array([[[  1.29282939e+00,   7.24226162e-02,   5.17048061e-01,
           8.93751144e-01],
        [  5.48077464e-01,  -1.90222517e-01,   4.40389782e-01,
           1.86340976e+00],
        [  4.28632259e-01,   1.25118005e+00,   4.43376899e-01,
           1.07255065e+00]],

       [[ -4.06714790e-02,   9.10973027e-02,   1.28347218e+00,
           1.03069496e+00],
        [  2.37148595e+00,   3.56667452e-02,   1.25722930e-01,
           1.17819163e-05],
        [  3.79356921e-01,   8.55060294e-02,   3.33660096e-02,
           3.40193957e-02]]], dtype=float32)
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

---
## **GaussianNoise**
Apply additive zero-centered Gaussian noise.

This is useful to mitigate overfitting (you could see it as a form of random data augmentation).

Gaussian Noise is a natural choice as corruption process for real valued inputs.

As it is a regularization layer, it is only active at training time.

**Scala:**
```scala
GaussianNoise(sigma, inputShape = null)
```
**Python:**
```python
GaussianNoise(sigma, input_shape=None, name=None)
```

**Parameters:**

* `sigma`: Standard deviation of the noise distribution.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.GaussianNoise
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GaussianNoise[Float](0.6, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.57896155     -0.19616802     1.7000706       -2.2136402
0.2245884       -0.167104       0.08521592      -0.31111532
-1.2676435      1.9858241       -0.27946314     -0.72280097

(2,.,.) =
1.263968        -0.1366611      0.7511876       -0.42096275
-0.2524562      -2.082302       -1.3312799      0.035666652
-1.6895409      -0.8562052      0.69322604      -0.080461726

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.25664312     0.1474515       2.066732        -1.5476861
0.34144306      1.1049318       0.4146787       -0.15529981
-1.3980585      2.0075183       0.09995845      -0.9865419

(2,.,.) =
0.8450401       0.0076646805    0.5062498       -0.5671178
0.89790833      -2.1620805      -1.5945435      -0.74607164
-1.7677919      -0.6946467      0.35671985      0.9388765

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import GaussianNoise
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(GaussianNoise(0.6, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
array([[[0.87836839, 0.29835789, 0.99199298, 0.61462649],
        [0.24045628, 0.9334569 , 0.69817451, 0.80795268],
        [0.82978091, 0.32160601, 0.97033687, 0.34726345]],

       [[0.11581215, 0.2012782 , 0.89101947, 0.24642749],
        [0.51231345, 0.47586449, 0.53419205, 0.71586367],
        [0.88794988, 0.20960408, 0.46741968, 0.31609195]]])
```
Output is
```python
array([[[ 0.9021132 ,  0.05798048,  0.9235187 ,  0.8105377 ],
        [ 0.82122934,  0.87509984,  1.3449373 ,  0.115228  ],
        [ 0.2612275 ,  0.02238336,  0.8971698 ,  0.3349191 ]],

       [[-0.7950512 , -0.4547084 ,  1.6517348 ,  1.5761411 ],
        [ 0.9232183 ,  0.33405185,  0.6043875 ,  0.54677534],
        [ 1.4350419 , -1.4409285 , -0.31246042,  0.5502143 ]]],
      dtype=float32)
```