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
