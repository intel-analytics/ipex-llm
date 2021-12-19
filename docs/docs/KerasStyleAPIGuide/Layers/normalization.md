## **BatchNormalization**
Batch normalization layer.

Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.

It is a feature-wise normalization, each feature map in the input will be normalized separately.

The input of this layer should be 4D.

**Scala:**
```scala
BatchNormalization(epsilon = 0.001, momentum = 0.99, betaInit = "zero", gammaInit = "one", dimOrdering = "th", inputShape = null)
```
**Python:**
```python
BatchNormalization(epsilon=0.001, momentum=0.99, beta_init="zero", gamma_init="one", dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `epsilon`: Fuzz parameter. Default is 0.001.
* `momentum`: Momentum in the computation of the exponential average of the mean and standard deviation of the data, for feature-wise normalization. Default is 0.99.
* `betaInit`: Name of initialization function for shift parameter. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'zero'.
* `gammaInit`: Name of initialization function for scale parameter. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'one'.
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'. For 'th', axis along which to normalize is 1. For 'tf', axis is 3.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.BatchNormalization
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(BatchNormalization[Float](inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.35774308      -0.0018262876   -1.0186636      -0.8283433
0.1458402       -0.8954456      0.65028995      0.74481136
0.46434486      -0.33841616     -0.2882468      0.27368018

(1,2,.,.) =
-0.85313565     -1.0957539      -0.7689828      1.7338694
0.66673565      1.0302666       -1.0154791      0.9704916
-1.518189       0.34307054      -0.8662138      0.53776205

(2,1,.,.) =
-1.5997988      0.4131082       -0.83005565     1.3930303
1.061352        -0.6628746      0.8510218       -0.36472544
1.4967325       -0.082105584    -1.2064567      0.5379558

(2,2,.,.) =
0.76886225      0.8283977       -2.815423       -1.1129401
-0.76033413     -0.09757436     -1.1177903      0.057090428
-1.1909146      1.3031846       1.8407855       2.2742975

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.42506456      -0.016198127    -1.2640586      -1.0304978
0.16501783      -1.1128457      0.7840774       0.9000738
0.55588603      -0.4292604      -0.3676927      0.32190278

(1,2,.,.) =
-0.66352594     -0.8604744      -0.59521383     1.4365083
0.57024884      0.86534977      -0.7953103      0.8168265
-1.2033914      0.30750957      -0.6741423      0.4655529

(2,1,.,.) =
-1.9772263      0.49300852      -1.0325992      1.6955665
1.2885318       -0.827435       1.030415        -0.4615471
1.8228296       -0.11471669     -1.4945178      0.6462212

(2,2,.,.) =
0.6531514       0.7014801       -2.2564375      -0.8744255
-0.5881931      -0.050189503    -0.8783628      0.0753616
-0.9377223      1.0868944       1.5232987       1.8752075

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import BatchNormalization

model = Sequential()
model.add(BatchNormalization(input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.90728825 0.06248136 0.38908736 0.41036892]
   [0.32752508 0.19828444 0.16125344 0.71703399]
   [0.91384765 0.10565062 0.5159064  0.11213003]]

  [[0.45955865 0.37912534 0.11220941 0.6227701 ]
   [0.74682518 0.31436052 0.35600359 0.46670668]
   [0.17039808 0.01137162 0.06768781 0.48850118]]]


 [[[0.41052004 0.51787735 0.22106962 0.72647921]
   [0.69059405 0.22422016 0.55071537 0.33162262]
   [0.92135018 0.81511106 0.76329409 0.30857876]]

  [[0.02103797 0.62061211 0.06155861 0.48460782]
   [0.95476727 0.66571869 0.53735588 0.09358965]
   [0.32302843 0.29893286 0.56494356 0.14670565]]]]
```
Output is
```python
[[[[ 1.5911555  -1.4893758  -0.2984292  -0.22082737]
   [-0.52291185 -0.9941791  -1.1292102   0.8974061 ]
   [ 1.6150738  -1.3319621   0.16400792 -1.3083354 ]]

  [[ 0.3420891   0.02168216 -1.0415802   0.99224377]
   [ 1.4864182  -0.2363091  -0.07042356  0.37056333]
   [-0.809785   -1.4432687  -1.2189325   0.45738205]]]


 [[[-0.2202763   0.17119484 -0.91109455  0.9318476 ]
   [ 0.8009946  -0.8996063   0.29093656 -0.5079704 ]
   [ 1.6424314   1.2550375   1.0660906  -0.59199834]]

  [[-1.4047626   0.98364717 -1.2433482   0.44187275]
   [ 2.314758    1.1633297   0.6519953  -1.1157522 ]
   [-0.20178048 -0.2977654   0.761891   -0.9041641 ]]]]
```

---
## **LRN2D**
Local Response Normalization between different feature maps.

**Scala:**
```scala
LRN2D(alpha = 1e-4, k = 1.0, beta = 0.75, n = 5, dimOrdering = "th", inputShape = null)
```
**Python:**
```python
LRN2D(alpha=1e-4, k=1.0, beta=0.75, n=5, dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `alpha`: Double. The scaling parameter. Default is 0.0001.
* `k`: Double. A constant. Default is 1.0.
* `beta`: Double. The exponent. Default is 0.75.
* `n`: The number of channels to sum over. Default is 5.
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.LRN2D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(LRN2D[Float](1e-3, 1.2, 0.4, 3, dimOrdering = "tf", inputShape = Shape(3, 3, 3)))
val input = Tensor[Float](2, 3, 3, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-0.6331058      -1.1622255      -0.20002009
0.031907756     1.4720777       0.36692062
0.16142464      -0.87992615     1.9201758

(1,2,.,.) =
-1.0693451      -1.0901353      0.6909652
0.13340907      1.0220904       -1.0232266
-1.4288133      0.8749622       -0.07012164

(1,3,.,.) =
-0.04984741     -1.4627954      1.2438095
1.5584376       -0.36223406     -0.862751
-0.68516856     -0.0066024275   -0.55539906

(2,1,.,.) =
1.8261654       -0.39168724     0.4531422
-0.09046966     0.61876625      0.4553172
0.58150214      -2.6587567      0.46114618

(2,2,.,.) =
0.75011647      -2.220607       -1.4024881
-0.5560173      0.19422908      -2.5069134
-0.7417007      1.3029631       -0.660577

(2,3,.,.) =
-0.17827246     1.8794266       1.2124214
0.5774041       0.25620413      0.6461205
0.33391082      -0.532468       1.3129597

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x3x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
-0.5884632      -1.0802679      -0.1859234
0.02965645      1.3681923       0.34102687
0.15005784      -0.81763095     1.7842402

(1,2,.,.) =
-0.9938776      -1.0131469      0.6422488
0.12401139      0.94998133      -0.95103925
-1.3279068      0.81316966      -0.065184206

(1,3,.,.) =
-0.046330474    -1.3593558      1.1558554
1.4484164       -0.33663353     -0.8019933
-0.63694555     -0.0061375294   -0.5163186

(2,1,.,.) =
1.6970686       -0.36398944     0.42125463
-0.08410302     0.5752084       0.4232657
0.54015917      -2.469669       0.4283661

(2,2,.,.) =
0.6969334       -2.0627165      -1.3028492
-0.5168911      0.18043552      -2.32896
-0.68936265     1.210961        -0.6139712

(2,3,.,.) =
-0.16566847     1.7462649       1.1265225
0.53676987      0.23816296      0.60064477
0.31041232      -0.49490157     1.2203434

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x3x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import LRN2D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(LRN2D(1e-3, 1.2, 0.4, 3, dim_ordering="tf", input_shape=(3, 3, 3)))
input = np.random.random([2, 3, 3, 3])
output = model.forward(input)
```
Input is:
```python
[[[[0.56356835, 0.57442602, 0.31515783],
   [0.64858065, 0.45682821, 0.63889742],
   [0.56114806, 0.32727298, 0.54948325]],

  [[0.25249933, 0.27872938, 0.2341261 ],
   [0.22254477, 0.0855324 , 0.95981825],
   [0.55280765, 0.722852  , 0.95902286]],

  [[0.65021279, 0.00722661, 0.64386904],
   [0.36467587, 0.84466816, 0.05716471],
   [0.16279813, 0.57831132, 0.52848513]]],


 [[[0.94372659, 0.32741784, 0.03196349],
   [0.06181632, 0.8300082 , 0.36091632],
   [0.4961609 , 0.5816011 , 0.95777095]],

  [[0.12676416, 0.32625023, 0.58114797],
   [0.05347868, 0.5303113 , 0.20170834],
   [0.76583324, 0.39418884, 0.84815322]],

  [[0.62523604, 0.56888912, 0.69009855],
   [0.34074716, 0.05078519, 0.05212047],
   [0.50672308, 0.30567418, 0.47902636]]]]
```
Output is
```python
[[[[0.5238933 , 0.53398067, 0.2929779 ],
   [0.602922  , 0.42464924, 0.59392124],
   [0.52165645, 0.30423048, 0.5108133 ]],

  [[0.23473667, 0.2591199 , 0.21765617],
   [0.20689127, 0.07950803, 0.8922195 ],
   [0.51387984, 0.6718813 , 0.89142925]],

  [[0.604453  , 0.00671771, 0.59855634],
   [0.3389953 , 0.7851862 , 0.05313992],
   [0.15134202, 0.53759885, 0.49128178]]],


 [[[0.87725437, 0.30435583, 0.02971505],
   [0.05746418, 0.77156085, 0.33550152],
   [0.46123454, 0.54060525, 0.89028406]],

  [[0.11784688, 0.30328864, 0.5402475 ],
   [0.04971581, 0.4929952 , 0.1875149 ],
   [0.7119114 , 0.36640498, 0.7884236 ]],

  [[0.58121526, 0.5288076 , 0.64150494],
   [0.31677726, 0.04721269, 0.04845466],
   [0.4710655 , 0.28415698, 0.44531912]]]]
```

---
## **WithinChannelLRN2D**
The local response normalization layer performs a kind of "lateral inhibition" by normalizing over local input regions. The local regions extend spatially, in separate channels (i.e., they have shape 1 x size x size).

**Scala:**
```scala
WithinChannelLRN2D(size = 5, alpha = 1.0, beta = 0.75, inputShape = null)
```
**Python:**
```python
WithinChannelLRN2D(size=5, alpha=1.0, beta=0.75, input_shape=None, name=None)
```

**Parameters:**

* `size`: The side length of the square region to sum over. Default is 5.
* `alpha`: The scaling parameter. Default is 1.0.
* `beta`: The exponent. Default is 0.75.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.WithinChannelLRN2D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(WithinChannelLRN2D[Float](inputShape = Shape(3, 4)))
val input = Tensor[Float](1, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.11547339     -0.52518076     0.22743009      0.24847448
-0.72996384     1.5127875       1.285603        -0.8665928
2.2911248       0.062601104     -0.07974513     -0.26207858

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.089576244    -0.39988548     0.17317083      0.21585277
-0.5662553      1.1518734       0.97888964      -0.7528196
1.7772957       0.047666013     -0.060719892    -0.22767082

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import WithinChannelLRN2D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(WithinChannelLRN2D(input_shape=(3, 4)))
input = np.random.random([1, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.96982874, 0.80581477, 0.35435895, 0.45537825],
  [0.61421818, 0.54708709, 0.86205409, 0.07374387],
  [0.67227822, 0.25118575, 0.36258901, 0.28671433]]]
```
Output is
```python
[[[0.87259495, 0.71950066, 0.3164021 , 0.42620906],
  [0.55263746, 0.48848635, 0.76971596, 0.06902022],
  [0.60487646, 0.22428022, 0.32375062, 0.26834887]]]
```
