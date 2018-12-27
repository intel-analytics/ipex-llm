## **Convolution1D**
Applies convolution operator for filtering neighborhoods of 1-D inputs.

You can also use `Conv1D` as an alias of this layer.

The input of this layer should be 3D.

**Scala:**
```scala
Convolution1D(nbFilter, filterLength, init = "glorot_uniform", activation = null, borderMode = "valid", subsampleLength = 1, wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
Convolution1D(nb_filter, filter_length, init="glorot_uniform", activation=None, border_mode="valid", subsample_length=1, W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

**Parameters:**

* `nbFilter`: Number of convolution filters to use.
* `filterLength`: The extension (spatial or temporal) of each filter.
* `init`: String representation of the initialization method for the weights of the layer. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'glorot_uniform'.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `subsampleLength`: Factor by which to subsample output. Integer. Default is 1.
* `wRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.Convolution1D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Convolution1D(8, 3, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-1.4253887	-0.044403594	-1.1169672	-0.19499049
0.85463065	0.6665206	    0.21340805	0.56255895
1.1126599	-0.3423326	    0.09643264	-0.34345046

(2,.,.) =
-0.04046587	-0.2710401	    0.10183265	1.4503858
1.0639644	1.5317003	    -0.18313104	-0.7098296
0.612399	1.7357533	    0.4641411	0.13530721

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.22175728	0.76192796	1.7907748	1.1534728	-1.5304534	0.07466106	-0.18292685	0.6038852

(2,.,.) =
0.85337734	0.43939286	-0.16770163	-0.8380078	0.7825804	-0.3485601	0.3017909	0.5823619

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Convolution1D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Convolution1D(8, 3, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.06092268 0.0508438  0.47256153 0.80004565]
  [0.48706905 0.65704781 0.04297214 0.42288264]
  [0.92286158 0.85394381 0.46423248 0.87896669]]

 [[0.216527   0.13880484 0.93482372 0.44812419]
  [0.95553331 0.27084259 0.58913626 0.01879454]
  [0.6656435  0.1985877  0.94133745 0.57504128]]]
```
Output is
```python
[[[ 0.7461933  -2.3189526  -1.454972   -0.7323345   1.5272427  -0.87963724  0.6278059  -0.23403725]]

 [[ 1.2397771  -0.9249111  -1.1432207  -0.92868984  0.53766745 -1.0271561  -0.9593589  -0.4768026 ]]]
```

---
## **LocallyConnected1D**
Locally-connected layer for 1D inputs which works similarly to the TemporalConvolution layer, except that weights are unshared, that is, a different set of filters is applied at each different patch of the input.

Border mode currently supported for this layer is 'valid'.

The input of this layer should be 3D.

**Scala:**
```scala
LocallyConnected1D(nbFilter, filterLength, activation = null, subsampleLength = 1, wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
LocallyConnected1D(nb_filter, filter_length, activation=None, border_mode="valid", subsample_length=1, W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

**Parameters:**

* `nbFilter`: Dimensionality of the output.
* `filterLength`: The extension (spatial or temporal) of each filter.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `subsampleLength`: Integer. Factor by which to subsample output.
* `wRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.LocallyConnected1D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(LocallyConnected1D(6, 3, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.6755046	0.47923228	-0.41470557	-1.4644535
-1.580751	-0.36924785	-1.1507624	0.20131736
-0.4983051	-2.0898817	0.1623063	0.8118141

(2,.,.) =
1.5955191	-1.1017833	1.6614468	1.7959124
1.1084127	0.528379	-1.114553	-1.030853
0.37758648	-2.5828059	1.0172523	-1.6773314

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.20011228	0.7842446	-0.57892114	0.2405633	-0.35126245	-0.5116563

(2,.,.) =
-0.33687726	0.7863857	0.30202985	0.33251244	-0.7414977	0.14271683

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1x6]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import LocallyConnected1D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(LocallyConnected1D(6, 3, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.67992353 0.88287213 0.98861104 0.17401607]
  [0.23660068 0.02779148 0.52982599 0.19876749]
  [0.38880073 0.6498778  0.81532701 0.91719509]]

 [[0.30532677 0.1574227  0.40535271 0.03174637]
  [0.37303714 0.27821415 0.02314422 0.64516966]
  [0.74813923 0.9884225  0.40667151 0.21894944]]]
```
Output is
```python
[[[ 0.66351205 -0.03819168 -0.48071918 -0.05209085 -0.07307816  0.94942856]]

 [[ 0.5890693   0.0179258  -0.31232932  0.4427027  -0.30954808  0.4486028 ]]]
```

---
## **UpSampling2D**
UpSampling layer for 2D inputs.

Repeats the rows and columns of the data by the specified size.

The input of this layer should be 4D.

**Scala:**
```scala
UpSampling2D(size = (2, 2), dimOrdering = "th", inputShape = null)
```
**Python:**
```python
UpSampling2D(size=(2, 2), dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `size`: Length 2. UpSampling factors for rows and columns. Default is (2, 2).
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.UpSampling2D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(UpSampling2D((2, 2), inputShape = Shape(2, 2, 2)))
val input = Tensor[Float](1, 2, 2, 2).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-0.07563081	-1.921836
-1.7368479	0.1043008

(1,2,.,.) =
-1.825055	-0.096810855
-0.89331573	0.72812295

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x2x2]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
-0.07563081	-0.07563081	 -1.921836	 -1.921836
-0.07563081	-0.07563081	 -1.921836	 -1.921836
-1.7368479	-1.7368479	 0.1043008	 0.1043008
-1.7368479	-1.7368479	 0.1043008	 0.1043008

(1,2,.,.) =
-1.825055	 -1.825055	  -0.096810855	-0.096810855
-1.825055	 -1.825055	  -0.096810855	-0.096810855
-0.89331573	 -0.89331573  0.72812295	0.72812295
-0.89331573	 -0.89331573  0.72812295	0.72812295

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x4x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import UpSampling2D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(UpSampling2D((2, 2), input_shape=(2, 2, 2)))
input = np.random.random([1, 2, 2, 2])
output = model.forward(input)
```
Input is:
```python
[[[[0.55660253 0.21984387]
   [0.36271854 0.57464162]]

  [[0.55307278 0.33007518]
   [0.31527167 0.87789644]]]]
```
Output is
```python
[[[[0.55660254 0.55660254 0.21984388 0.21984388]
   [0.55660254 0.55660254 0.21984388 0.21984388]
   [0.36271855 0.36271855 0.57464164 0.57464164]
   [0.36271855 0.36271855 0.57464164 0.57464164]]

  [[0.55307275 0.55307275 0.33007517 0.33007517]
   [0.55307275 0.55307275 0.33007517 0.33007517]
   [0.31527168 0.31527168 0.8778964  0.8778964 ]
   [0.31527168 0.31527168 0.8778964  0.8778964 ]]]]
```

---
## **UpSampling3D**
UpSampling layer for 3D inputs.

Repeats the 1st, 2nd and 3rd dimensions of the data by the specified size.

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

The input of this layer should be 5D.

**Scala:**
```scala
UpSampling3D(size = (2, 2, 2), dimOrdering = "th", inputShape = null)
```
**Python:**
```python
UpSampling3D(size=(2, 2, 2), dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `size`: Length 3. UpSampling factors for dim1, dim2 and dim3. Default is (2, 2, 2).
* `dimOrdering`: Format of input data. Only 'th' (Channel First) is supported for now.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.UpSampling3D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(UpSampling3D[Float]((2, 2, 2), inputShape = Shape(1, 1, 2, 2)))
val input = Tensor[Float](1, 1, 1, 2, 2).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
0.05876646      0.8743367
-0.15551122     0.9405281

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x1x2x2]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
0.05876646      0.05876646      0.8743367       0.8743367
0.05876646      0.05876646      0.8743367       0.8743367
-0.15551122     -0.15551122     0.9405281       0.9405281
-0.15551122     -0.15551122     0.9405281       0.9405281

(1,1,2,.,.) =
0.05876646      0.05876646      0.8743367       0.8743367
0.05876646      0.05876646      0.8743367       0.8743367
-0.15551122     -0.15551122     0.9405281       0.9405281
-0.15551122     -0.15551122     0.9405281       0.9405281

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x2x4x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import UpSampling3D

model = Sequential()
model.add(UpSampling3D((2, 2, 2), input_shape=(1, 1, 2, 2)))
input = np.random.random([1, 1, 1, 2, 2])
output = model.forward(input)
```
Input is:
```python
[[[[[0.01897243 0.87927954]
-    [0.13656585 0.3003842 ]]]]]
```
Output is
```python
[[[[[0.01897243 0.01897243 0.87927955 0.87927955]
-    [0.01897243 0.01897243 0.87927955 0.87927955]
-    [0.13656585 0.13656585 0.3003842  0.3003842 ]
-    [0.13656585 0.13656585 0.3003842  0.3003842 ]]
-
-   [[0.01897243 0.01897243 0.87927955 0.87927955]
-    [0.01897243 0.01897243 0.87927955 0.87927955]
-    [0.13656585 0.13656585 0.3003842  0.3003842 ]
-    [0.13656585 0.13656585 0.3003842  0.3003842 ]]]]]
```

---
## **AtrousConvolution1D**
Applies an atrous convolution operator for filtering neighborhoods of 1-D inputs.

A.k.a dilated convolution or convolution with holes.

Bias will be included in this layer.

Border mode currently supported for this layer is 'valid'.

You can also use `AtrousConv1D` as an alias of this layer.

The input of this layer should be 3D.

**Scala:**
```scala
AtrousConvolution1D(nbFilter, filterLength, init = "glorot_uniform", activation = null, subsampleLength = 1, atrousRate = 1, wRegularizer = null, bRegularizer = null, inputShape = null)
```
**Python:**
```python
AtrousConvolution1D(nb_filter, filter_length, init="glorot_uniform", activation=None, border_mode='valid', subsample_length=1, atrous_rate=1, W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

**Parameters:**

* `nbFilter`: Number of convolution kernels to use.
* `filterLength`: The extension (spatial or temporal) of each filter.
* `init`: String representation of the initialization method for the weights of the layer. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'glorot_uniform'.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `subsampleLength`: Factor by which to subsample output. Integer. Default is 1.
* `atrousRate`: Factor for kernel dilation. Also called filter_dilation elsewhere. Integer. Default is 1.
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.AtrousConvolution1D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(AtrousConvolution1D[Float](8, 3, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.18186663     -0.43034658     0.26391524      -1.4132749
-0.17445838     1.3798479       0.1737039       1.152537
0.27590567      0.009284354     -0.80261934     -0.9434588

(2,.,.) =
-0.20791245     0.21988653      0.8744776       0.2940677
0.07080339      0.51823103      -0.46097854     -0.037812505
0.35226902      0.79622966      0.011483789     0.88822025

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.026210725     1.2229221       0.45232815      -1.0826558      0.849349        0.086645454     0.041758537     0.3721839

(2,.,.) =
-0.14264873     0.060507685     -0.217965       0.42317814      0.17935039      -0.05465065     -0.6533742      -0.009769946

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import AtrousConvolution1D

model = Sequential()
model.add(AtrousConvolution1D(8, 3, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.44706076 0.5902202  0.3784323  0.4098717 ]
  [0.74646876 0.98997355 0.64164388 0.61591103]
  [0.88695659 0.16591123 0.6575717  0.55897158]]

 [[0.51990872 0.82065542 0.18409799 0.99078291]
  [0.03853884 0.0781884  0.82290244 0.99992993]
  [0.02394716 0.10870804 0.17077537 0.77893951]]]
```
Output is
```python
[[[-0.09361145  0.48225394 -0.3777458  -0.84651476  0.3678655
   -0.02871403  1.0220621   0.7548751 ]]

 [[-0.0299319   0.37761992 -0.08759689 -0.01757497 -0.01414538
   -0.2547227   0.70025307  0.49045497]]]
```