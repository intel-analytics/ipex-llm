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
## **AtrousConvolution2D**
Applies an atrous convolution operator for filtering windows of 2-D inputs.

A.k.a dilated convolution or convolution with holes.

Bias will be included in this layer.

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

Border mode currently supported for this layer is 'valid'.

You can also use `AtrousConv2D` as an alias of this layer.

The input of this layer should be 4D.

**Scala:**
```scala
AtrousConvolution2D(nbFilter, nbRow, nbCol, init = "glorot_uniform", activation = null, subsample = (1, 1), atrousRate= (1, 1), dimOrdering = "th", wRegularizer = null, bRegularizer = null, inputShape = null)
```
**Python:**
```python
AtrousConvolution2D(nb_filter, nb_row, nb_col, init="glorot_uniform", activation=None, border_mode="valid", subsample=(1, 1), atrous_rate=(1, 1), dim_ordering="th", W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

**Parameters:**

* `nbFilter`: Number of convolution filters to use.
* `nbRow`: Number of rows in the convolution kernel.
* `nbCol`: Number of columns in the convolution kernel.
* `init`: String representation of the initialization method for the weights of the layer. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'glorot_uniform'.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `subsample`: Length 2 . Factor by which to subsample output. Also called strides elsewhere. Default is (1, 1).
* `atrousRate`: Length 2. Factor for kernel dilation. Also called filter_dilation elsewhere. Default is (1, 1).
* `dimOrdering`: Format of input data. Only 'th' (Channel First) is supported for now.
* `wRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), applied to the bias. Default is null.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.AtrousConvolution2D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(AtrousConvolution2D[Float](4, 2, 2, activation = "relu", inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-0.57626903	    -0.56916714	    0.46516004	    -1.189643
-0.117406875	-1.1139084	    1.115328	    0.23275337
1.452733	    -0.30968842	    -0.6693723	    -0.22098665

(1,2,.,.) =
0.06541251	    -0.7000564	    -0.460471	    -0.5291468
-0.6625642	    0.6460361	    -0.556095	    1.6327276
1.1914377	    -0.69054496	    -0.7461783	    -1.0129389

(2,1,.,.) =
-0.19123174	    0.06803144	    -0.010993495	-0.79966563
-0.010654963	2.0806832	    1.972848	    -1.8525643
-0.84387285	    1.2974458	    -0.42781293	    0.3540522

(2,2,.,.) =
1.6231914	    0.52689505	    0.47506556	    -1.030227
0.5143046	    -0.9930063	    -2.2869735	    0.03994834
-1.5566326	    -1.0937842	    0.82693833	    -0.08408405

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.11401264	0.0	        1.1396459
0.0	        0.0	        0.88493514

(1,2,.,.) =
0.0	        8.398667	1.1495202
0.0	        0.0	        0.1630742

(1,3,.,.) =
0.0	        0.92470163	0.0
0.0	        0.6321572	0.0

(1,4,.,.) =
0.0	        1.1912066	0.0
0.0	        1.27647	    0.13422263

(2,1,.,.) =
0.0	        0.0	        0.51365596
0.0	        0.4207713	1.1226959

(2,2,.,.) =
0.0	        0.67600054	0.63635653
0.40892223	2.0596464	1.7690754

(2,3,.,.) =
1.1899394	0.0	        0.0
1.7185769	0.39178902	0.0

(2,4,.,.) =
0.44333076	0.73385376	0.0
2.516453	0.36223468	0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import AtrousConvolution2D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(AtrousConvolution2D(4, 2, 2, input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.52102612 0.30683086 0.38543426 0.0026452 ]
   [0.66805249 0.60656045 0.94601998 0.46574414]
   [0.49391338 0.14274225 0.70473703 0.30427041]]
  [[0.89066007 0.51782675 0.7063052  0.53440807]
   [0.67377917 0.51541465 0.02137767 0.63357007]
   [0.6614106  0.15849977 0.94459604 0.46852022]]]

 [[[0.79639026 0.94468413 0.73165819 0.54531867]
   [0.97741046 0.64477619 0.52373183 0.06861999]
   [0.37278645 0.53198045 0.95098245 0.86249644]]
  [[0.47186038 0.81694951 0.78009033 0.20925898]
   [0.69942883 0.37150324 0.58907364 0.88754231]
   [0.64083971 0.4480097  0.91716521 0.66808943]]]]
```
Output is
```python
[[[[-0.32139003  -0.34667802 -0.35534883]
   [-0.09653517  -0.35052428 -0.09859636]]
  [[-0.3138999   -0.5563417  -0.6694119 ]
   [-0.03151364  0.35521197  0.31497604]]
  [[-0.34939283  -0.7537081  -0.3939833 ]
   [-0.25708836  0.06015673  -0.16755156]]
  [[-0.04791902  0.02060626  -0.5639752 ]
   [ 0.16054101  0.22528952  -0.02460545]]]

 [[[-0.13129832  -0.5262137   -0.12281597]
   [-0.36988598  -0.5532047   -0.43338764]]
  [[-0.21627764  -0.17562683  0.23560521]
   [ 0.23035726  -0.03152001  -0.46413773]]
  [[-0.63740283  -0.33359224  0.15731882]
   [-0.12795202  -0.25798583  -0.5261132 ]]
  [[-0.01759483  -0.07666921  -0.00890112]
   [ 0.27595833  -0.14117064  -0.3357542 ]]]]
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
model.add(UpSampling2D[Float]((2, 2), inputShape = Shape(2, 2, 2)))
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

---
## **ZeroPadding1D**
Zero-padding layer for 1D input (e.g. temporal sequence).

The input of this layer should be 3D.

**Scala:**
```scala
ZeroPadding1D(padding = 1, inputShape = null)
```
**Python:**
```python
ZeroPadding1D(padding=1, input_shape=None, name=None)
```

**Parameters:**

* `padding`: How many zeros to add at the beginning and at the end of the padding dimension.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.ZeroPadding1D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(ZeroPadding1D[Float](1, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.7421485	-0.13270181	-0.12605186	-0.7442475
0.36977226	-0.90300065	-0.34193754	-0.035565257
-0.23300397	0.8183156	0.7023575	-0.16938858

(2,.,.) =
-0.7785278	0.36642975	-1.0542017	-0.29036212
-0.22632122	0.46808097	-0.68293047	1.2529073
-0.8619831	1.3846883	1.0762612	1.1351995

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.0	        0.0	        0.0	        0.0
0.7421485	-0.13270181	-0.12605186	-0.7442475
0.36977226	-0.90300065	-0.34193754	-0.035565257
-0.23300397	0.8183156	0.7023575	-0.16938858
0.0	        0.0	        0.0	        0.0

(2,.,.) =
0.0	        0.0	        0.0	        0.0
-0.7785278	0.36642975	-1.0542017	-0.29036212
-0.22632122	0.46808097	-0.68293047	1.2529073
-0.8619831	1.3846883	1.0762612	1.1351995
0.0	        0.0	        0.0	        0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x5x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import ZeroPadding1D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(ZeroPadding1D(1, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.74177145 0.75805981 0.2091588  0.46929227]
  [0.46041743 0.13213793 0.51065024 0.36081853]
  [0.60803218 0.27764702 0.31788482 0.65445294]]

 [[0.96255443 0.74692762 0.50050961 0.88456158]
  [0.55492653 0.50850271 0.17788885 0.91569285]
  [0.27356035 0.74622588 0.39690752 0.75229177]]]
```
Output is
```python
[[[0.0        0.0        0.0        0.0       ]
  [0.74177146 0.7580598  0.2091588  0.46929225]
  [0.46041742 0.13213794 0.5106502  0.36081854]
  [0.60803217 0.27764702 0.31788483 0.6544529 ]
  [0.0        0.0        0.0        0.0       ]]

 [[0.0        0.0        0.0        0.0       ]
  [0.96255445 0.7469276  0.5005096  0.8845616 ]
  [0.5549265  0.5085027  0.17788884 0.91569287]
  [0.27356035 0.7462259  0.39690754 0.75229174]
  [0.0        0.0        0.0        0.0       ]]]
```

---
## **ZeroPadding3D**
Zero-padding layer for 3D data (spatial or spatio-temporal).

The input of this layer should be 5D.

**Scala:**
```scala
ZeroPadding3D(padding = (1, 1, 1), dimOrdering = "th", inputShape = null)
```
**Python:**
```python
ZeroPadding3D(padding=(1, 1, 1), dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `padding`: Int array of length 3. How many zeros to add at the beginning and end of the 3 padding dimensions. Symmetric padding will be applied to each dimension. Default is (1, 1, 1).
* `dimOrdering`: Format of the input data. Either "CHANNEL_FIRST" (dimOrdering='th') or "CHANNEL_LAST" (dimOrdering='tf'). Default is "CHANNEL_FIRST".
* `inputShape`: A Single Shape, does not include the batch dimension.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.ZeroPadding3D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(ZeroPadding3D[Float](padding = (1, 1, 1), inputShape = Shape(1, 2, 1, 2)))
val input = Tensor[Float](1, 1, 2, 1, 2).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
-0.59840345     -0.06308561

(1,1,2,.,.) =
0.48804763      0.2723002

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x2x1x2]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
0.0     0.0     0.0     0.0
0.0     0.0     0.0     0.0
0.0     0.0     0.0     0.0

(1,1,2,.,.) =
0.0     0.0     0.0     0.0
0.0     -0.59840345     -0.06308561     0.0
0.0     0.0     0.0     0.0

(1,1,3,.,.) =
0.0     0.0     0.0     0.0
0.0     0.48804763      0.2723002       0.0
0.0     0.0     0.0     0.0

(1,1,4,.,.) =
0.0     0.0     0.0     0.0
0.0     0.0     0.0     0.0
0.0     0.0     0.0     0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x4x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import ZeroPadding3D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(ZeroPadding3D(padding=(1, 1, 1), input_shape=(1, 2, 1, 2)))
input = np.random.random([1, 1, 2, 1, 2])
output = model.forward(input)
```
Input is:
```python
[[[[[0.03167021, 0.15764403]],

   [[0.26572586, 0.48872052]]]]]
```
Output is
```python
[[[[[0.        , 0.        , 0.        , 0.        ],
    [0.        , 0.        , 0.        , 0.        ],
    [0.        , 0.        , 0.        , 0.        ]],

   [[0.        , 0.        , 0.        , 0.        ],
    [0.        , 0.03167021, 0.15764403, 0.        ],
    [0.        , 0.        , 0.        , 0.        ]],

   [[0.        , 0.        , 0.        , 0.        ],
    [0.        , 0.26572585, 0.48872054, 0.        ],
    [0.        , 0.        , 0.        , 0.        ]],

   [[0.        , 0.        , 0.        , 0.        ],
    [0.        , 0.        , 0.        , 0.        ],
    [0.        , 0.        , 0.        , 0.        ]]]]]
```
