## **GaussianSampler**
Takes {mean, log_variance} as input and samples from the Gaussian distribution.

When you use this layer as the first layer of a model, you need to provide the argument inputShape (a Single Shape, does not include the batch dimension).

**Scala:**
```scala
GaussianSampler(inputShape = null)
```
**Python:**
```python
GaussianSampler(input_shape=None, name=None)
```

**Parameters:**

* `inputShape`: For Scala API, it is a single Shape, does not include the batch dimension.
For Python API, it should be a list of shape. 

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.GaussianSampler
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GaussianSampler[Float](inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-0.38246885     -0.42231864     -0.6192802      -0.3716393
-0.3531316      1.3189775       0.41730723      2.0870953
1.0675176       -0.4296848      0.48483646      0.6121345

(1,2,.,.) =
-0.41407543     0.052751783     -0.25280526     -0.4470164
-0.08855078     1.8301593       -0.2534143      0.3276937
-0.8874503      -0.3305329      -1.4175192      -0.15308261

(2,1,.,.) =
-1.9688231      -0.66597825     1.2758924       0.4109092
-0.3716698      -1.3192323      0.32637826      -0.84648186
-0.98650885     -1.0351782      -0.01710262     0.37902358

(2,2,.,.) =
-0.3126414      0.11948945      -0.30405575     0.8048093
-0.52251136     -2.6021087      0.11725392      -0.29351738
0.75019157      0.17965068      -0.8849312      1.1649832

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
-0.38246885     -0.42231864     -0.6192802      -0.3716393
-0.3531316      1.3189775       0.41730723      2.0870953
1.0675176       -0.4296848      0.48483646      0.6121345

(1,2,.,.) =
-0.41407543     0.052751783     -0.25280526     -0.4470164
-0.08855078     1.8301593       -0.2534143      0.3276937
-0.8874503      -0.3305329      -1.4175192      -0.15308261

(2,1,.,.) =
-1.9688231      -0.66597825     1.2758924       0.4109092
-0.3716698      -1.3192323      0.32637826      -0.84648186
-0.98650885     -1.0351782      -0.01710262     0.37902358

(2,2,.,.) =
-0.3126414      0.11948945      -0.30405575     0.8048093
-0.52251136     -2.6021087      0.11725392      -0.29351738
0.75019157      0.17965068      -0.8849312      1.1649832

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import GaussianSampler

model = Sequential()
model.add(GaussianSampler(input_shape=[(3,),(3,)]))
input1 = np.random.random([2, 3])
input2 = np.random.random([2, 3])
input = [input1, input2]
output = model.forward(input)
```
Input is:
```python
[array([[0.79941342, 0.87462822, 0.9516901 ],
       [0.20111287, 0.54634077, 0.83614511]]), array([[0.31886989, 0.22829382, 0.84355419],
       [0.51186641, 0.28043938, 0.29440057]])]
```
Output is
```python
[[ 0.71405387  2.2944303  -0.41778684]
 [ 0.84234     2.3337283  -0.18952972]]
```

---
## **Exp**
Applies element-wise exp to the input.

When you use this layer as the first layer of a model, you need to provide the argument inputShape (a Single Shape, does not include the batch dimension).

**Scala:**
```scala
Exp(inputShape = null)
```
**Python:**
```python
Exp(input_shape=None, name=None)
```

**Parameters:**

* `inputShape`: A Single Shape, does not include the batch dimension.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.Exp
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Exp[Float](inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-1.5841372      -0.13795324     -2.144475       0.09272669
1.055668        -1.2310301      1.2145554       -0.6073714
0.9296467       0.2923885       1.3364213       0.1652137

(1,2,.,.) =
0.2099718       -0.3856573      -0.92586        -0.5317779
0.6618383       -0.9677452      -1.5014665      -0.35464883
2.045924        -0.317644       -1.812726       0.95438373

(2,1,.,.) =
-0.4536791      -0.34785584     1.6424289       -0.07981159
-0.8022624      -0.4211059      0.3461831       1.9598864
-0.84695745     -0.6115283      0.7729755       2.3077402

(2,2,.,.) =
-0.08438411     -0.908458       0.6688936       -0.7292123
-0.26337254     0.55425745      -0.14925817     -0.010179609
-0.62562865     -1.0517743      -0.23839666     -1.144982

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.20512469      0.8711394       0.11712951      1.0971619
2.8738942       0.29199165      3.3687959       0.544781
2.533614        1.3396233       3.8054006       1.1796452

(1,2,.,.) =
1.2336433       0.6800035       0.39619055      0.5875594
1.9383523       0.37993878      0.22280318      0.7014197
7.7363033       0.7278619       0.16320862      2.5970695

(2,1,.,.) =
0.63528657      0.70620066      5.167706        0.92329025
0.44831353      0.6563206       1.4136615       7.0985208
0.42871734      0.5425211       2.1662023       10.051684

(2,2,.,.) =
0.9190782       0.4031454       1.9520763       0.48228875
0.76845556      1.740648        0.8613467       0.98987204
0.53492504      0.34931743      0.7878901       0.31822965

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import Exp

model = Sequential()
model.add(Exp(input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.93104587 0.94000338 0.84870765 0.98645553]
   [0.83708846 0.33375541 0.50119834 0.24879265]
   [0.51966475 0.84514791 0.15496452 0.61538968]]

  [[0.57250337 0.42520832 0.94850757 0.54317573]
   [0.64228691 0.9904079  0.01008592 0.51365217]
   [0.78640595 0.7717037  0.51277595 0.24245034]]]


 [[[0.82184752 0.92537331 0.20632728 0.47539445]
   [0.44604637 0.1507692  0.5437313  0.2074501 ]
   [0.93661363 0.93962609 0.29230559 0.74850958]]

  [[0.11659768 0.76177132 0.33194573 0.20695088]
   [0.49636212 0.85987328 0.49767861 0.96774006]
   [0.67669121 0.15542122 0.69981032 0.3349874 ]]]]
```
Output is
```python
[[[[2.5371614 2.5599902 2.3366253 2.6817122]
   [2.3096325 1.3962016 1.6506982 1.2824761]
   [1.6814638 2.3283222 1.1676165 1.8503776]]

  [[1.7726992 1.5299091 2.5818534 1.721465 ]
   [1.9008229 2.6923325 1.010137  1.6713842]
   [2.1954916 2.163449  1.6699204 1.2743679]]]


 [[[2.2746985 2.52281   1.2291554 1.6086487]
   [1.5621239 1.1627283 1.7224218 1.2305363]
   [2.551327  2.5590243 1.3395122 2.1138473]]

  [[1.1236672 2.1420672 1.3936772 1.2299222]
   [1.6427343 2.3628614 1.6448984 2.6319895]
   [1.9673574 1.16815   2.0133708 1.3979228]]]]
```

---
## **Square**
Applies an element-wise square operation to the input.

When you use this layer as the first layer of a model, you need to provide the argument inputShape (a Single Shape, does not include the batch dimension).

**Scala:**
```scala
Square(inputShape = null)
```
**Python:**
```python
Square(input_shape=None, name=None)
```

**Parameters:**

* `inputShape`: A Single Shape, does not include the batch dimension.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.Square
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Square[Float](inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-0.108013034    1.8879265       1.2232096       -1.5076439
1.4895755       -0.37966672     -0.34892964     0.15224025
-0.9296686      -1.1523775      0.14153497      -0.26954007

(1,2,.,.) =
-1.0875931      2.190617        -0.6903083      1.0039362
-0.1275677      -1.1096588      0.37359753      -0.17367937
0.23349741      0.14639114      -0.2330162      0.5343827

(2,1,.,.) =
0.3222191       0.21463287      -1.0157064      -0.22627507
1.1714277       0.43371263      1.069315        0.5122436
0.1958086       -1.4601041      2.5394423       -0.470833

(2,2,.,.) =
-0.38708544     -0.951611       -0.37234613     0.26813275
1.9477026       0.32779223      -1.2308712      -2.2376378
0.19652915      0.3304719       -1.7674786      -0.86961496

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.011666816     3.5642662       1.4962418       2.2729902
2.218835        0.14414681      0.1217519       0.023177093
0.86428374      1.3279738       0.020032147     0.07265185

(1,2,.,.) =
1.1828587       4.7988033       0.47652552      1.0078878
0.016273517     1.2313428       0.13957511      0.030164523
0.05452104      0.021430366     0.054296546     0.28556487

(2,1,.,.) =
0.10382515      0.046067268     1.0316595       0.05120041
1.3722429       0.18810664      1.1434345       0.26239353
0.038341008     2.131904        6.448767        0.22168371

(2,2,.,.) =
0.14983514      0.9055635       0.13864164      0.07189517
3.7935455       0.10744774      1.5150439       5.007023
0.038623706     0.109211676     3.1239805       0.7562302

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import Square

model = Sequential()
model.add(Square(input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.8708819  0.2698243  0.55854849 0.71699472]
   [0.66647234 0.72310216 0.8082119  0.66566951]
   [0.6714764  0.61394108 0.35063125 0.60473593]]

  [[0.37993365 0.64222557 0.96762005 0.18931697]
   [0.00529722 0.99133455 0.09786619 0.28988077]
   [0.60052911 0.83712995 0.59847519 0.54361243]]]


 [[[0.32832672 0.83316023 0.41272485 0.01963383]
   [0.89593955 0.73433713 0.67529323 0.69711912]
   [0.81251711 0.56755577 0.31958151 0.09795917]]

  [[0.46465895 0.22818875 0.31505317 0.41912166]
   [0.87865447 0.3799063  0.091204   0.68144165]
   [0.88274284 0.70479132 0.32074672 0.71771481]]]]
```
Output is
```python
[[[[7.5843531e-01 7.2805151e-02 3.1197643e-01 5.1408142e-01]
   [4.4418535e-01 5.2287674e-01 6.5320653e-01 4.4311589e-01]
   [4.5088059e-01 3.7692365e-01 1.2294226e-01 3.6570552e-01]]

  [[1.4434958e-01 4.1245368e-01 9.3628860e-01 3.5840917e-02]
   [2.8060573e-05 9.8274422e-01 9.5777912e-03 8.4030852e-02]
   [3.6063525e-01 7.0078653e-01 3.5817260e-01 2.9551446e-01]]]


 [[[1.0779844e-01 6.9415593e-01 1.7034180e-01 3.8548734e-04]
   [8.0270761e-01 5.3925103e-01 4.5602092e-01 4.8597506e-01]
   [6.6018403e-01 3.2211956e-01 1.0213234e-01 9.5959986e-03]]

  [[2.1590793e-01 5.2070107e-02 9.9258497e-02 1.7566296e-01]
   [7.7203369e-01 1.4432879e-01 8.3181690e-03 4.6436274e-01]
   [7.7923489e-01 4.9673077e-01 1.0287846e-01 5.1511449e-01]]]]
```