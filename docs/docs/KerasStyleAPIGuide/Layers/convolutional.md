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
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Convolution1D}
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
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Convolution1D

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
## **Convolution2D**
Applies a 2D convolution over an input image composed of several input planes.

You can also use `Conv2D` as an alias of this layer.

The input of this layer should be 4D.

**Scala:**
```scala
Convolution2D(nbFilter, nbRow, nbCol, init = "glorot_uniform", activation = null, borderMode = "valid", subsample = (1, 1), dimOrdering = "th", wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
Convolution2D(nb_filter, nb_row, nb_col, init="glorot_uniform", activation=None, border_mode="valid", subsample=(1, 1), dim_ordering="th", W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

**Parameters:**

* `nbFilter`: Number of convolution filters to use.
* `nbRow`: Number of rows in the convolution kernel.
* `nbCol`: Number of columns in the convolution kernel.
* `init`: String representation of the initialization method for the weights of the layer. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'glorot_uniform'.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `subsample`: Length 2 corresponding to the step of the convolution in the height and width dimension. Also called strides elsewhere. Default is (1, 1).
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Convolution2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Convolution2D(4, 2, 2, activation = "relu", inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-1.6687597	1.3452173	-1.9608531	-0.30892205
1.7459077	-0.443617	0.25789636	0.44496542
-1.5395774	-0.37713575	-0.9973955	0.16208267

(1,2,.,.) =
0.593965	1.0544858	-1.0765858	0.22257836
0.69452614	1.3700147	-0.886259	0.013910895
-1.9819669	0.32151425	1.8303248	0.24231844

(2,1,.,.) =
-2.150859	-1.5894475	0.7543173	0.7713991
-0.17536041	0.89053404	0.50489277	-0.098128
0.11551995	1.3663125	0.76734704	0.28318745

(2,2,.,.) =
-0.9801306	0.39797616	-0.6403248	1.0090133
-0.16866015	-1.426308	-2.4097774	0.26011375
-2.5700948	1.0486397	-0.4585798	-0.94231766

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.0	        0.6800046	0.0
1.0366663	0.235024	0.0

(1,2,.,.) =
0.0	        0.84696645	0.0
0.9099177	0.0	        0.0

(1,3,.,.) =
0.122891426	0.0	        0.86579126
0.0	        0.0	        0.0

(1,4,.,.) =
0.0	        0.7185988	0.0
1.0967548	0.48376864	0.0

(2,1,.,.) =
0.0	        0.0	        0.29164955
0.06815311	0.0	        0.0

(2,2,.,.) =
0.36370438	0.42393038	0.26948324
1.1676859	0.5698308	0.44842285

(2,3,.,.) =
1.0797265	1.2410768	0.18289843
0.0	        0.0	        0.18757495

(2,4,.,.) =
0.0	        0.35713753	0.0
0.0	        0.0	        0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Convolution2D

model = Sequential()
model.add(Convolution2D(4, 2, 2, input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.05182015 0.91971256 0.81030852 0.64093699]
   [0.60282957 0.16269049 0.79136121 0.05202386]
   [0.62560999 0.00174107 0.75762976 0.93589574]]
  [[0.13728558 0.85812609 0.39695457 0.81678788]
   [0.7569393  0.61161632 0.60750583 0.6222684 ]
   [0.53094821 0.38715199 0.0087283  0.05758945]]]

 [[[0.50030948 0.40179766 0.54900785 0.60950401]
   [0.17464329 0.01506322 0.55273153 0.21567461]
   [0.09037649 0.58831638 0.818708   0.96642448]]
  [[0.77126628 0.58039509 0.91612417 0.12578268]
   [0.6095838  0.15802154 0.78099004 0.63619778]
   [0.70632951 0.91378968 0.84851605 0.7242516 ]]]]
```
Output is
```python
[[[[-0.45239753  0.2432243  -0.02717562]
   [ 0.2698849  -0.09664132  0.92311716]]
  [[ 0.9092748   0.7945191   0.8834159 ]
   [ 0.4853364   0.6511425   0.52513427]]
  [[ 0.5550465   0.8177169   0.43213058]
   [ 0.4209347   0.7514105   0.27255327]]
  [[-0.22105691  0.02853963  0.01092601]
   [ 0.1258291  -0.41649136 -0.18953061]]]

 [[[-0.12111888  0.06418754  0.26331317]
   [ 0.41674113  0.04221775  0.7313505 ]]
  [[ 0.49442202  0.6964868   0.558412  ]
   [ 0.25196168  0.8145504   0.69307953]]
  [[ 0.5885831   0.59289575  0.71726865]
   [ 0.46759683  0.520353    0.59305453]]
  [[ 0.00594708  0.09721318  0.07852311]
   [ 0.49868047  0.02704304  0.14635414]]]]
```

---
## **Convolution3D**
Applies convolution operator for filtering windows of three-dimensional inputs.

You can also use `Conv3D` as an alias of this layer.

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

The input of this layer should be 5D.

**Scala:**
```scala
Convolution3D(nbFilter, kernelDim1, kernelDim2, kernelDim3, init = "glorot_uniform", activation = null, borderMode = "valid", subsample = (1, 1, 1), dimOrdering = "th", wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
Convolution3D(nb_filter, kernel_dim1, kernel_dim2, kernel_dim3, init="glorot_uniform", activation=None, border_mode="valid", subsample=(1, 1, 1), dim_ordering="th", W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

**Parameters:**

* `nbFilter`: Number of convolution filters to use.
* `kernelDim1`: Length of the first dimension in the convolution kernel.
* `kernelDim2`: Length of the second dimension in the convolution kernel.
* `kernelDim3`: Length of the third dimension in the convolution kernel.
* `init`: String representation of the initialization method for the weights of the layer. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'glorot_uniform'.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `subsample`: Length 3. Factor by which to subsample output. Also called strides elsewhere. Default is (1, 1, 1).
* `dimOrdering`: Format of input data. Only 'th' (Channel First) is supported for now.
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Convolution3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Convolution3D(4, 2, 2, 2, inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
0.64140224	-1.4049392	0.4935015
0.33096266	1.2768826	-0.57567996

(1,1,2,.,.) =
0.49570087	-2.0367618	  -0.0032108661
-0.24242361	-0.092683665  1.1579652

(1,2,1,.,.) =
-0.6730608	0.9149566	-1.7478822
-0.1763675	-0.90117735	0.38452747

(1,2,2,.,.) =
0.5314353	1.4802488	-1.196325
0.43506134	-0.56575996	-1.5489199

(2,1,1,.,.) =
0.074545994	-1.4092928	-0.57647055
1.9998664	-0.19424418	-0.9296713

(2,1,2,.,.) =
-0.42966184	0.9247804	-0.21713361
0.2723336	-1.3024703	1.278154

(2,2,1,.,.) =
1.1240695	1.1061385	-2.4662287
-0.36022148	0.1620907	-1.1525819

(2,2,2,.,.) =
0.9885768	-0.526637	-0.40684605
0.37813842	0.53998697	1.0001947

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
0.36078936	-0.6334647

(1,2,1,.,.) =
-0.19685572	0.4559337

(1,3,1,.,.) =
1.3750207	-2.4377227

(1,4,1,.,.) =
-0.82853335	-0.74145436

(2,1,1,.,.) =
-0.17973013	1.2930126

(2,2,1,.,.) =
0.69144577	0.44385013

(2,3,1,.,.) =
-0.5597819	0.5965629

(2,4,1,.,.) =
0.89755684	-0.6737796

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x1x1x2]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Convolution3D

model = Sequential()
model.add(Convolution3D(4, 2, 2, 2, input_shape=(2, 2, 2, 3)))
input = np.random.random([2, 2, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[[[0.36798873 0.92635561 0.31834968]
    [0.09001392 0.66762381 0.64477164]]
   [[0.09760993 0.38067132 0.15069965]
    [0.39052699 0.0223722  0.04786307]]]
  [[[0.2867726  0.3674255  0.11852931]
    [0.96436629 0.8012903  0.3211012 ]]
   [[0.81738622 0.80606827 0.4060485 ]
    [0.68010177 0.0934071  0.98479026]]]]

 [[[[0.71624597 0.37754442 0.07367964]
    [0.60742428 0.38549046 0.78880978]]
   [[0.97844361 0.11426373 0.55479659]
    [0.06395313 0.86007246 0.34004405]]]
  [[[0.94149643 0.8027673  0.19478027]
    [0.17437108 0.754479   0.51055297]]
   [[0.81933677 0.09040694 0.33775061]
    [0.02582059 0.40027544 0.91809986]]]]]

```
Output is
```python
[[[[[ 1.6276866  -4.4106215]]]
  [[[ 6.6988254  1.1409638]]]
  [[[-5.7734865  -5.2575850]]]
  [[[-1.8073934  -4.4056013]]]]

 [[[[-4.8580116  9.4352424]]]
  [[[ 7.8890514  6.6063654]]]
  [[[-7.3165756  -1.0116580]]]
  [[[-1.3100024  1.0475740]]]]]
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
import com.intel.analytics.bigdl.nn.keras.{Sequential, AtrousConvolution1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(AtrousConvolution1D(8, 3, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.18167362	2.1452308	-0.39164552	-0.19750737
-0.16184713	-1.3867316	-1.3447738	-0.6431075
-0.42635638	-0.20490816	-2.5391808	-0.05881459

(2,.,.) =
-0.83197606	1.1706954	-0.80197126	-1.0663458
0.36859998	-0.45194706	-1.2959619	-0.521925
-1.133602	0.7700087	-1.2523394	1.1293458

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.1675637	0.9712032	-0.5615059	0.065867506	0.6681816	1.2097323	-1.0912716	0.8040266

(2,.,.) =
0.5009172	1.4765333	-0.14173388	0.060548827	0.752389	1.2912648	-1.0077878	0.06344204

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1x8]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import AtrousConvolution1D

model = Sequential()
model.add(AtrousConvolution1D(8, 3, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.96952262 0.15211776 0.63026888 0.50572335]
  [0.13218867 0.18807126 0.33509675 0.43385223]
  [0.48027981 0.82222524 0.9630902  0.78855421]]

 [[0.49106312 0.16875464 0.54099084 0.2892753 ]
  [0.03776569 0.51324722 0.95359981 0.52863015]
  [0.69851295 0.29676433 0.59404524 0.90078511]]]
```
Output is
```python
[[[-0.62304074  0.6667814  -0.07074605 -0.03640915  0.11369559  0.3451041  -0.44238544  0.618591  ]]

 [[-0.5048915   0.9070808  -0.03285386  0.26761323 -0.08491824  0.36105093 -0.15240929  0.6145356 ]]]
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
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, AtrousConvolution2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(AtrousConvolution2D(4, 2, 2, activation = "relu", inputShape = Shape(2, 3, 4)))
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
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import AtrousConvolution2D

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
## **Deconvolution2D**
Transposed convolution operator for filtering windows of 2-D inputs.

The need for transposed convolutions generally arises from the desire to use a transformation going in the opposite direction of a normal convolution, i.e., from something that has
the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution.

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

Border mode currently supported for this layer is 'valid'.

You can also use `Deconv2D` as an alias of this layer.

The input of this layer should be 4D.

**Scala:**
```scala
Deconvolution2D(nbFilter, nbRow, nbCol, init = "glorot_uniform", activation = null, subsample = (1, 1), dimOrdering = "th", wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
Deconvolution2D(nb_filter, nb_row, nb_col, output_shape, init="glorot_uniform", activation=None, border_mode="valid", subsample=(1, 1), dim_ordering="th", W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

**Parameters:**

* `nbFilter`: Number of transposed convolution filters to use.
* `nbRow`: Number of rows in the transposed convolution kernel.
* `nbCol`: Number of columns in the transposed convolution kernel.
* `init`: String representation of the initialization method for the weights of the layer. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'glorot_uniform'.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `subsample`: Length 2 . The step of the convolution in the height and width dimension. Also called strides elsewhere. Default is (1, 1).
* `dimOrdering`: Format of input data. Only 'th' (Channel First) is supported for now.
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Deconvolution2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Deconvolution2D(2, 2, 2, activation = "relu", inputShape = Shape(2, 2, 3)))
val input = Tensor[Float](1, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-1.1157457	-0.8626509	-0.7326707
1.8340882	-1.1647098	-1.0159439

(1,2,.,.) =
-0.13360074	0.4507607	-0.5922559
0.15494606	0.16541296	1.6870573

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.0	        0.0	        0.0	    0.020009547
0.0	        0.0	        0.0	    0.0
0.9656998	0.0	        0.0	    0.5543601

(1,2,.,.) =
0.0	        0.0	        0.0	    0.07773054
1.4971795	0.029338006	0.0	    0.0
0.0	        0.45826393	0.0	    0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Deconvolution2D

model = Sequential()
model.add(Deconvolution2D(2, 2, 2, (2, 3, 4), input_shape=(2, 2, 3)))
input = np.random.random([1, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[[0.65315139 0.21904901 0.57943617]
   [0.35141043 0.14628658 0.81862311]]

  [[0.60094717 0.84649884 0.08338504]
   [0.26753695 0.83676038 0.87466877]]]]
```
Output is
```python
[[[[-0.35380065  0.22048733  0.3084591   0.23341973]
   [-0.11611718  0.5349988  -0.26301163  1.0291481 ]
   [ 0.00479569  0.48814884  0.00127316  0.2546792 ]]

  [[-0.02683929 -0.21759698 -0.8542665  -0.25376737]
   [ 0.04426606  0.05486238 -0.9282576  -1.1576774 ]
   [ 0.01637976  0.1838439  -0.01419228 -0.60704494]]]]
```

---
## **SeparableConvolution2D**
Applies separable convolution operator for 2D inputs.

Separable convolutions consist in first performing a depthwise spatial convolution (which acts on each input channel separately) followed by a pointwise convolution which mixes together the resulting output channels. The depthMultiplier argument controls how many output channels are generated per input channel in the depthwise step.

You can also use `SeparableConv2D` as an alias of this layer.

The input of this layer should be 4D.

**Scala:**
```scala
SeparableConvolution2D(nbFilter, nbRow, nbCol, init = "glorot_uniform", activation = null, borderMode = "valid", subsample = (1, 1), depthMultiplier = 1, dimOrdering = "th", depthwiseRegularizer = null, pointwiseRegularizer= null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
SeparableConvolution2D(nb_filter, nb_row, nb_col, init="glorot_uniform", activation=None, border_mode="valid", subsample=(1, 1), depth_multiplier=1, dim_ordering="th", depthwise_regularizer=None, pointwise_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

**Parameters:**

* `nbFilter`: Number of convolution filters to use.
* `nbRow`: Number of rows in the convolution kernel.
* `nbCol`: Number of columns in the convolution kernel.
* `init`: String representation of the initialization method for the weights of the layer. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'glorot_uniform'.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `subsample`: Length 2 corresponding to the step of the convolution in the height and width dimension. Also called strides elsewhere. Default is (1, 1).
* `depthMultiplier`: How many output channel to use per input channel for the depthwise convolution step. Integer. Default is 1.
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `depthwiseRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the depthwise weights matrices. Default is null.
* `pointwiseRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the pointwise weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, SeparableConvolution2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(SeparableConvolution2D(2, 2, 2, activation = "relu", inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.61846036	0.13724488	   1.9047198	0.8788536
0.74383116	-0.7590018	   0.17210509	1.8095028
-0.21476124	-0.010768774   0.5437478	0.97470677

(1,2,.,.) =
-0.22464052	-1.7141389	   1.8457758	0.81563693
-0.17250067	-1.2183974	   -2.5329974	-1.3014348
0.43760046	0.32672745	   -0.6059157	0.31439257

(2,1,.,.) =
-0.32413644	-0.1871411	   -0.13821407	-0.16577224
-0.02138366	1.2260025	   -0.48404458	-1.0251912
-1.8844653	0.6796752	   -0.5881143	2.1656246

(2,2,.,.) =
0.17234507	-0.6455974	   1.9615031	0.6552883
-0.05861185	1.8847446	   -0.857622	-0.5949971
-0.41135395	-0.92089206	   0.13154007	-0.9326055

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.0	        0.0	        0.0
0.029595211	0.0	        0.34002993

(1,2,.,.) =
0.0	        0.0	        0.0
0.073145226	0.0	        0.5542682

(2,1,.,.) =
0.4973382	0.36478913	0.0
0.0	        0.0	        0.0

(2,2,.,.) =
0.9668598	0.7102739	0.0
0.0	        0.0	        0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import SeparableConvolution2D

model = Sequential()
model.add(SeparableConvolution2D(2, 2, 2, input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.39277921 0.36904141 0.16768533 0.41712068]
   [0.62416696 0.19334139 0.83341541 0.16486488]
   [0.57287259 0.47809379 0.11103843 0.01746644]]
  [[0.24945342 0.05728102 0.19076369 0.70498077]
   [0.39147172 0.08100018 0.74426575 0.74251056]
   [0.61840056 0.00771785 0.65170218 0.04492181]]]

 [[[0.08337509 0.19320791 0.66757918 0.38905916]
   [0.50237454 0.0996316  0.3981495  0.32274897]
   [0.01598124 0.52896577 0.76068351 0.10099803]]
  [[0.20396797 0.48682425 0.11302674 0.57491998]
   [0.71529612 0.11720466 0.57783092 0.45790133]
   [0.41573101 0.60269287 0.613528   0.32717263]]]]
```
Output is
```python
[[[[0.15971108 0.12109925 0.17461367]
   [0.20024002 0.13661252 0.1871847 ]]
  [[0.47139192 0.36838844 0.45902973]
   [0.57752806 0.41371965 0.5079273 ]]]

 [[[0.11111417 0.10702941 0.2030398 ]
   [0.13108528 0.15029006 0.18544158]]
  [[0.27002305 0.31479427 0.57750916]
   [0.3573216  0.40100253 0.5122235 ]]]]
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
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, LocallyConnected1D}
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
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import LocallyConnected1D

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
## **LocallyConnected2D**
Locally-connected layer for 2D inputs that works similarly to the SpatialConvolution layer, except that weights are unshared, that is, a different set of filters is applied at each different patch of the input.

The input of this layer should be 4D.

**Scala:**
```scala
LocallyConnected2D(nbFilter, nbRow, nbCol, activation = null, borderMode = "valid", subsample = (1, 1), dimOrdering = "th", wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
LocallyConnected2D(nb_filter, nb_row, nb_col, activation=None, border_mode="valid", subsample=(1, 1), dim_ordering="th", W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

**Parameters:**

* `nbFilter`: Number of convolution filters to use.
* `nbRow`: Number of rows in the convolution kernel.
* `nbCol`: Number of columns in the convolution kernel.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `subsample`: Length 2 corresponding to the step of the convolution in the height and width dimension. Also called strides elsewhere. Default is (1, 1).
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, LocallyConnected2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(LocallyConnected2D(2, 2, 2, inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
1.3119988	    -1.8982307	    -0.13138956	    1.0872058
-0.11329581	    -0.7087005	    0.085274234	    -0.94051
1.04928	        2.1579344	    -1.4412278	    -0.90965116

(1,2,.,.) =
-0.6119555	    1.2226686	    -0.10441754	    -1.6240023
0.5598073	    -0.099059306	-1.543586	    0.72533834
-1.6674699	    -1.0901593	    -0.24129404	    0.30954796

(2,1,.,.) =
-0.78856885	    -0.5567014	    -1.1273636	    -0.98069143
-0.40949664	    0.92562497	    -1.3729718	    0.7423901
-0.29498738	    -0.044669412	1.0937366	    0.90768206

(2,2,.,.) =
1.0948726	    -0.23575573	    -0.051821854	-0.58692485
1.9133459	    -1.0849183	    2.1423934	    0.6559134
-0.8390565	    -0.27111387	    -0.8439365	    -1.3939567

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
-0.42428172	0.25790718	-0.5227444
0.6963143	-0.34605533	-0.35524538

(1,2,.,.) =
0.61758286	0.8430548	0.1378907
0.24116383	0.15782532	0.16882366

(2,1,.,.) =
-0.5603108	0.5107949	-0.112701565
0.62288725	0.6909297	-0.9253155

(2,2,.,.) =
-0.2443612	0.9310517	-0.2417406
-0.82973266	-1.0886648	0.19112866

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import LocallyConnected2D

model = Sequential()
model.add(LocallyConnected2D(2, 2, 2, input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.57424593 0.49505236 0.63711108 0.43693806]
   [0.34655799 0.0058394  0.69310344 0.70403367]
   [0.4620432  0.58679338 0.64529398 0.78130808]]
  [[0.49651564 0.32201482 0.02470762 0.80535793]
   [0.94485185 0.07150504 0.58789497 0.4562848 ]
   [0.63595033 0.04600271 0.89771801 0.95419454]]]

 [[[0.69641827 0.21785002 0.15815588 0.8317213 ]
   [0.84192366 0.3939658  0.64309395 0.3858968 ]
   [0.16545408 0.58533897 0.99486481 0.84651898]]
  [[0.05144159 0.94930242 0.26842063 0.6341632 ]
   [0.442836   0.38544902 0.04266468 0.22600452]
   [0.2705393  0.07313841 0.24295287 0.9573069 ]]]]
```
Output is
```python
[[[[ 0.1600316   0.178018   -0.07472821]
   [ 0.0570091  -0.19973318  0.44483435]]
  [[-0.20258084 -0.37692443 -0.27103102]
   [-0.624092   -0.09749079 -0.00799894]]]

 [[[ 0.58953685  0.35287908 -0.2203412 ]
   [ 0.13649486 -0.29554832  0.16932982]]
  [[-0.00787066 -0.06614903 -0.2027885 ]
   [-0.33434835 -0.33458236 -0.15103136]]]]
```

---
## **Cropping1D**
Cropping layer for 1D input (e.g. temporal sequence).

The input of this layer should be 3D.

**Scala:**
```scala
Cropping1D(cropping = (1, 1), inputShape = null)
```
**Python:**
```python
Cropping1D(cropping=(1, 1), input_shape=None, name=None)
```

**Parameters:**

* `cropping`: Length 2. How many units should be trimmed off at the beginning and end of the cropping dimension. Default is (1, 1).
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Cropping1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Cropping1D((1, 1), inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-1.0038188	-0.75265634	0.7417358	1.0674809
-1.4702164	0.64112693	0.17750219	-0.21439286
-0.93766433	-1.0809567	0.7706962	0.16380796

(2,.,.) =
0.45019576	-0.36689326	0.08852628	-0.21602148
0.66039973	0.11638404	0.062985964	-1.0420738
0.46727908	-0.85894865	1.9853845	0.059447426

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-1.4702164	0.64112693	0.17750219	-0.21439286

(2,.,.) =
0.66039973	0.11638404	0.062985964	-1.0420738

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Cropping1D

model = Sequential()
model.add(Cropping1D(input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.01030651 0.77603525 0.97263208 0.15933375]
  [0.05135971 0.01139832 0.28809891 0.57260363]
  [0.28128354 0.55290954 0.77011153 0.09879061]]

 [[0.75765909 0.55102462 0.42426818 0.14383546]
  [0.85198966 0.3990277  0.13061313 0.10349525]
  [0.69892804 0.30310119 0.2241441  0.05978997]]]
```
Output is
```python
[[[0.05135971 0.01139832 0.2880989  0.57260364]]

 [[0.8519896  0.3990277  0.13061313 0.10349525]]]
```

---
## **Cropping2D**
Cropping layer for 2D input (e.g. picture).

The input of this layer should be 4D.

**Scala:**
```scala
Cropping2D(cropping = ((0, 0), (0, 0)), dimOrdering = "th", inputShape = null)
```
**Python:**
```python
Cropping2D(cropping=((0, 0), (0, 0)), dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `cropping`: Int tuple of tuple of length 2. How many units should be trimmed off at the beginning and end of the 2 cropping dimensions (i.e. height and width). Default is ((0, 0), (0, 0)).
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Cropping2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Cropping2D(((0, 1), (1, 0)), inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-1.3613406	-0.03520738	 -0.008660733  2.1150143
0.18087284	1.8787018	 0.30097032	  -2.5634677
-1.9463011	-0.18772388	 1.5215846	  -0.8047026

(1,2,.,.) =
-0.50510925	-1.1193116	 0.6901347	 -0.2625669
-0.24307655	-0.77917117	 -0.566465	 1.0432123
0.4877474	0.49704018	 -1.5550427	 1.5772455

(2,1,.,.) =
-1.6180872	0.011832007	 1.2762135	 0.5600022
1.9009352	-0.11096256	 1.1500957	 -0.26341736
1.0153246	0.88008636	 0.0560876	 -1.0235065

(2,2,.,.) =
0.1036221	1.08527	     -0.52559805   -0.5091204
1.3085281	-0.96346164	 -0.09713245   -1.1010116
0.08505145	1.9413263	 2.0237558	   -0.5978173

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
-0.03520738	-0.008660733  2.1150143
1.8787018	0.30097032	  -2.5634677

(1,2,.,.) =
-1.1193116	 0.6901347	-0.2625669
-0.77917117	 -0.566465	1.0432123

(2,1,.,.) =
0.011832007	 1.2762135	0.5600022
-0.11096256	 1.1500957	-0.26341736

(2,2,.,.) =
1.08527	     -0.52559805   -0.5091204
-0.96346164	 -0.09713245   -1.1010116

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Cropping2D

model = Sequential()
model.add(Cropping2D(((0, 1), (1, 0)), input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.03691489 0.60233732 0.96327319 0.99561146]
   [0.85728883 0.77923287 0.41328434 0.87490199]
   [0.3389653  0.94804499 0.72922732 0.21191413]]
  [[0.28962322 0.30133445 0.58516862 0.22476588]
   [0.95386045 0.72488497 0.12056255 0.01265548]
   [0.48645173 0.34426033 0.09410422 0.86815053]]]

 [[[0.57444115 0.79141167 0.20755353 0.38616465]
   [0.95793123 0.22366943 0.5080078  0.27193368]
   [0.65402317 0.1023231  0.67207896 0.2229965 ]]
  [[0.04160647 0.55577895 0.30907277 0.42227706]
   [0.54489229 0.90423796 0.50782414 0.51441165]
   [0.87544565 0.47791071 0.0341273  0.14728084]]]]
```
Output is
```python
[[[[0.6023373  0.96327317 0.9956115 ]
   [0.77923286 0.41328433 0.874902  ]]
  [[0.30133444 0.5851686  0.22476588]
   [0.724885   0.12056255 0.01265548]]]

 [[[0.7914117  0.20755354 0.38616467]
   [0.22366942 0.5080078  0.27193367]]

  [[0.555779   0.30907276 0.42227706]
   [0.904238   0.5078241  0.5144116 ]]]]
```

---
## **Cropping3D**
Cropping layer for 3D data (e.g. spatial or spatio-temporal).

The input of this layer should be 5D.

**Scala:**
```scala
Cropping3D(cropping = ((1, 1), (1, 1), (1, 1)), dimOrdering = "th", inputShape = null)
```
**Python:**
```python
Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `cropping`: Int tuple of tuple of length 3. How many units should be trimmed off at the beginning and end of the 3 cropping dimensions (i.e. kernel_dim1, kernel_dim2 and kernel_dim3). Default is ((1, 1), (1, 1), (1, 1)).
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Cropping3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Cropping3D(((1, 1), (1, 1), (1, 1)), inputShape = Shape(2, 3, 4, 5)))
val input = Tensor[Float](2, 2, 3, 4, 5).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
-0.12339484	    0.25661087	    0.04387503	    -1.1047344	    -1.1413815
1.1830065	    -0.07189157	    -0.5418846	    0.5576781	    -0.5460917
-0.5679186	    -0.30854696	    1.2614665	    -0.6774269	    -0.63295823
0.5269464	    -2.7981617	    -0.056265026	-1.0814936	    -1.0848739

(1,1,2,.,.) =
-1.9100302	    0.461067	    0.4014941	    0.60723174	    -0.40414023
0.34300476	    0.7107094	    1.3142885	    1.5696589	    0.97591686
0.38320687	    0.07036536	    -0.43628898	    0.58050656	    -0.57882625
-0.43699506	    -0.0094956765	0.15171598	    0.038076796	    -1.2433665

(1,1,3,.,.) =
0.39671394	    0.880047	    0.30971292	    -0.3369089	    0.13062176
-0.27803114 	-0.62177086	    0.16659822	    0.89428085	    0.23684736
1.6151237	    -1.1479733	    -0.2229254	    1.1361892	    0.79478127
-1.8207864	    1.6544164	    0.07977915	    -1.1316417	    -0.25483203

(1,2,1,.,.) =
1.3165517	    -0.9479057	    -1.4662051	    -0.3343554	    -0.4522552
-1.5829691	    0.6378519	    -0.16399206	    1.4724066	    1.2387054
-1.1467208	    -0.6325814	    -1.2106491	    -0.035734158	0.19871919
2.285004	    1.0482147	    -2.0056705  	-0.80917794	    2.523167

(1,2,2,.,.) =
-0.57108706	    -0.23606259	    -0.45569882	    -0.034214735    -1.9130942
-0.2743481	    1.61177	        -0.7052599	    0.17889105	    -0.31241596
0.22377247	    1.5860337	    -0.3226252	    -0.1341058	    0.9239994
0.03615294	    0.6233593	    0.757827	    -0.72271305	    0.9429943

(1,2,3,.,.) =
-0.4409662	    0.8867786	    2.0036085	    0.16242673	    -0.3332395
0.09082064	    0.04958198	    -0.27834833	    1.8025815	    -0.04848101
0.2690667	    -1.1263227	    -0.95486647	    0.09473259	    0.98166656
-0.9509363	    -0.10084029	    -0.35410827	    0.29626986	    0.97203517

(2,1,1,.,.) =
0.42096403	    0.14016314	    0.20216857	    -0.678293	    -1.0970931
-0.4981112	    0.12429344	    1.7156922	    -0.24384527 	-0.010780937
0.03672217	    2.3021698	    1.568247	    -0.43173146	    -0.5550057
0.30469602	    1.4772439	    -0.21195345 	0.04221814	    -1.6883365

(2,1,2,.,.) =
0.22468264	    0.72787744	    -0.9597003	    -0.28472963	    -1.4575284
1.0487963	    0.4982454	    -1.0186157	    -1.9877508	    -1.133779
0.17539643	    -0.35151628	    -1.8955303	    2.1854792	    0.59556997
0.6893949	    -0.19556235	    0.25862908	    0.24450152	    0.17786922

(2,1,3,.,.) =
1.147159	    -0.8849993	    0.9826487	    0.95360875	    -0.9210176
1.3439047	    0.6739913	    0.06558858	    0.91963255  	-1.1758618
1.747105	    -0.7225308	    -1.0160877	    0.67554474	    -0.7762811
0.21184689	    -0.43668815	    -1.0738864	    0.04661594	    0.9613895

(2,2,1,.,.) =
-0.377159	    -0.28094378	    0.1081715	    1.3683178       1.2572801
0.47781375	    0.4545212	    0.55356956	    1.0366637	    -0.1962683
-1.820227	    -0.111765414	1.9194998	    -1.6089902	    -1.6960226
0.14896627	    0.9360371	    0.49156702	    0.08601956	    -0.08815153

(2,2,2,.,.) =
0.056315728	    -0.13061485	    -0.49018836	    -0.59103477     -1.6910721
-0.023719765	-0.44977355	    0.11218439	    0.224829	    1.400084
0.31496882	    -1.6386473	    -0.6715097	    0.14816228	    0.3240011
-0.80607724	    -0.37951842	    -0.2187672	    1.1087769	    0.43044603

(2,2,3,.,.) =
-1.6647842	    -0.5720825	    -1.5150099	    0.42346838	    1.495052
-0.3567161	    -1.4341534	    -0.19422509	    -1.2871891	    -1.2758921
-0.47077888	    -0.42217267	    0.67764246	    1.2170314	    0.8420698
-0.4263702	    1.2792329	    0.38645822	    -2.4653213	    -1.512707

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4x5]

```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
0.7107094	1.3142885	1.5696589
0.07036536	-0.43628898	0.58050656

(1,2,1,.,.) =
1.61177	    -0.7052599	0.17889105
1.5860337	-0.3226252	-0.1341058

(2,1,1,.,.) =
0.4982454	-1.0186157	-1.9877508
-0.35151628	-1.8955303	2.1854792

(2,2,1,.,.) =
-0.44977355	0.11218439	0.224829
-1.6386473	-0.6715097	0.14816228

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x1x2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Cropping3D

model = Sequential()
model.add(Cropping3D(((1, 1), (1, 1), (1, 1)), input_shape=(2, 3, 4, 5)))
input = np.random.random([2, 2, 3, 4, 5])
output = model.forward(input)
```
Input is:
```python
[[[[[0.17398425 0.68189365 0.77769123 0.53108205 0.64715435]
    [0.14553671 0.56312657 0.68612354 0.69176945 0.30109699]
    [0.09732807 0.37460879 0.19945361 0.86471357 0.66225896]
    [0.23071766 0.7940814  0.20828491 0.05256511 0.39059369]]
   [[0.61604377 0.08752888 0.0373393  0.2074062  0.60620641]
    [0.72873275 0.86871873 0.89248703 0.9407502  0.71830713]
    [0.23277175 0.75968678 0.2160847  0.76278034 0.27796526]
    [0.45593022 0.31406512 0.83030059 0.17528758 0.56134316]]
   [[0.65576189 0.41055457 0.90979203 0.76003643 0.26369912]
    [0.20767533 0.60489496 0.44996379 0.20016757 0.39282226]
    [0.14055952 0.15767185 0.70149107 0.88403803 0.77345544]
    [0.34344548 0.03721154 0.86204782 0.45349481 0.69348787]]]
  [[[0.55441874 0.59949813 0.4450893  0.2103161  0.6300366 ]
    [0.71573331 0.32423206 0.06302588 0.91902299 0.30852669]
    [0.73540519 0.20697542 0.20543135 0.44461869 0.89286638]
    [0.41614996 0.48155318 0.51663767 0.23681825 0.34780746]]
   [[0.34529962 0.81156897 0.77911935 0.65392321 0.45178564]
    [0.39702465 0.36180668 0.37867952 0.24818676 0.84365902]
    [0.67836434 0.24043224 0.59870659 0.81976809 0.95442206]
    [0.15342281 0.48607751 0.11420129 0.68621285 0.09892679]]
   [[0.61122758 0.40359022 0.99805441 0.76764677 0.6281926 ]
    [0.44867213 0.81206033 0.40117858 0.98967612 0.76897064]
    [0.90603977 0.17299288 0.68803644 0.75164168 0.4161878 ]
    [0.18996933 0.93317759 0.77711184 0.50760022 0.77439241]]]]

 [[[[0.49974828 0.74486599 0.12447392 0.15415173 0.36715309]
    [0.49334423 0.66699219 0.22202136 0.52689596 0.15497081]
    [0.4117844  0.21886979 0.13096058 0.82589121 0.00621519]
    [0.38257617 0.60924058 0.53549974 0.64299846 0.66315369]]
   [[0.78048895 0.20350694 0.16485496 0.71243727 0.4581091 ]
    [0.554526   0.66891789 0.90082079 0.76729771 0.40647459]
    [0.72809646 0.68164733 0.83008334 0.90941546 0.1441997 ]
    [0.44580521 0.78015871 0.63982938 0.26813225 0.15588673]]
   [[0.85294056 0.0928758  0.37056251 0.82930655 0.27178195]
    [0.95953427 0.60170629 0.69156911 0.27902576 0.55613879]
    [0.97101437 0.49876892 0.36313494 0.11233855 0.24221145]
    [0.28739626 0.2990425  0.68940864 0.95621615 0.6922569 ]]]
  [[[0.90283303 0.51320503 0.78356741 0.79301195 0.17681709]
    [0.61624755 0.95418399 0.68118889 0.69241549 0.17943311]
    [0.71129437 0.55478761 0.34121912 0.86018439 0.03652437]
    [0.39098173 0.87916544 0.39647239 0.00104663 0.01377085]]
   [[0.28875017 0.03733266 0.47260498 0.2896268  0.55976704]
    [0.08723092 0.45523634 0.98463086 0.56950302 0.98261442]
    [0.20716971 0.52744283 0.39455719 0.57384754 0.76698272]
    [0.3079253  0.88143353 0.85897125 0.0969679  0.43760548]]
   [[0.44239165 0.56141652 0.30344311 0.05425044 0.34003295]
    [0.31417344 0.39485584 0.47300811 0.38006721 0.23185974]
    [0.06158527 0.95330693 0.63043506 0.9480669  0.93758737]
    [0.05340179 0.2064604  0.97254971 0.60841205 0.89738937]]]]]
```
Output is
```python
[[[[[0.86871874 0.89248705 0.9407502 ]
    [0.75968677 0.2160847  0.7627803 ]]]
  [[[0.3618067  0.3786795  0.24818675]
    [0.24043223 0.5987066  0.8197681 ]]]]

 [[[[0.6689179  0.9008208  0.7672977 ]
    [0.68164736 0.8300834  0.9094155 ]]]
  [[[0.45523635 0.9846309  0.569503  ]
    [0.5274428  0.39455718 0.57384753]]]]]
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
import com.intel.analytics.bigdl.nn.keras.{Sequential, ZeroPadding1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(ZeroPadding1D(1, inputShape = Shape(3, 4)))
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
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import ZeroPadding1D

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
## **ZeroPadding2D**
Zero-padding layer for 2D input (e.g. picture).

The input of this layer should be 4D.

**Scala:**
```scala
ZeroPadding2D(padding = (1, 1), dimOrdering = "th", inputShape = null)
```
**Python:**
```python
ZeroPadding2D(padding=(1, 1), dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `padding`: How many zeros to add at the beginning and at the end of the 2 padding dimensions (rows and cols).
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, ZeroPadding2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(ZeroPadding2D((1, 1), inputShape = Shape(2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
1.7201442	-1.0197405	1.3163399
-0.23921064	0.7732504	-0.069928266

(1,2,.,.) =
0.46323594	-1.3043984	-0.67622787
-1.610615	-0.39253974	-0.89652705

(2,1,.,.) =
-0.3784847	-0.6738694	0.30479854
-0.49577644	1.0704983	0.6288544

(2,2,.,.) =
0.2821439	0.790223	0.34665197
0.24190207	0.10775433	0.46225727

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.0	    0.0	        0.0	        0.0	            0.0
0.0	    1.7201442	-1.0197405	1.3163399	    0.0
0.0	    -0.23921064	0.7732504	-0.069928266	0.0
0.0	    0.0	        0.0	        0.0	            0.0

(1,2,.,.) =
0.0	    0.0	        0.0	        0.0	            0.0
0.0	    0.46323594	-1.3043984	-0.67622787	    0.0
0.0	    -1.610615	-0.39253974	-0.89652705	    0.0
0.0	    0.0	        0.0	        0.0	            0.0

(2,1,.,.) =
0.0	    0.0	        0.0	        0.0	            0.0
0.0	    -0.3784847	-0.6738694	0.30479854	    0.0
0.0	    -0.49577644	1.0704983	0.6288544	    0.0
0.0	    0.0	        0.0	        0.0	            0.0

(2,2,.,.) =
0.0	    0.0	        0.0	        0.0	            0.0
0.0	    0.2821439	0.790223	0.34665197	    0.0
0.0	    0.24190207	0.10775433	0.46225727	    0.0
0.0	    0.0	        0.0	        0.0	            0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x4x5]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import ZeroPadding2D

model = Sequential()
model.add(ZeroPadding2D(input_shape=(2, 2, 3)))
input = np.random.random([2, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[[0.44048214 0.72494886 0.96654241]
   [0.66254801 0.37409083 0.47681466]]

  [[0.23204026 0.52762765 0.15072852]
   [0.45052127 0.29016392 0.0133929 ]]]


 [[[0.09347565 0.4754528  0.63618458]
   [0.08016674 0.21696158 0.83892852]]

  [[0.81864575 0.90813398 0.08347963]
   [0.57234761 0.76060611 0.65707858]]]]
```
Output is
```python
[[[[0.0   0.0        0.0        0.0        0.0 ]
   [0.0   0.44048214 0.7249489  0.9665424  0.0 ]
   [0.0   0.662548   0.37409082 0.47681466 0.0 ]
   [0.0   0.0        0.0        0.0        0.0 ]]

  [[0.0   0.0        0.0        0.0        0.0 ]
   [0.0   0.23204026 0.52762765 0.15072852 0.0 ]
   [0.0   0.45052126 0.29016393 0.0133929  0.0 ]
   [0.0   0.0        0.0        0.0        0.0 ]]]


 [[[0.0   0.0        0.0        0.0        0.0 ]
   [0.0   0.09347565 0.4754528  0.6361846  0.0 ]
   [0.0   0.08016673 0.21696158 0.8389285  0.0 ]
   [0.0   0.0        0.0        0.0        0.0 ]]

  [[0.0   0.0        0.0        0.0        0.0 ]
   [0.0   0.8186458  0.908134   0.08347963 0.0 ]
   [0.0   0.5723476  0.7606061  0.65707856 0.0 ]
   [0.0   0.0        0.0        0.0        0.0 ]]]]
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

* `padding`: How many zeros to add at the beginning and end of the 3 padding dimensions. Symmetric padding will be applied to each dimension.
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, ZeroPadding3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(ZeroPadding3D((1, 1, 1), inputShape = Shape(1, 2, 2, 2)))
val input = Tensor[Float](1, 1, 2, 2, 2).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
1.086798	2.162806
-0.50501716	-0.17430544

(1,1,2,.,.) =
-1.7388326	0.27966997
1.6211525	1.1713351

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x2x2x2]

```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
0.0	    0.0	        0.0	        0.0
0.0	    0.0	        0.0	        0.0
0.0	    0.0	        0.0	        0.0
0.0	    0.0	        0.0	        0.0

(1,1,2,.,.) =
0.0	    0.0	        0.0	        0.0
0.0	    1.086798	2.162806	0.0
0.0	    -0.50501716	-0.17430544	0.0
0.0	    0.0	        0.0	        0.0

(1,1,3,.,.) =
0.0	    0.0	        0.0	        0.0
0.0	    -1.7388326	0.27966997	0.0
0.0	    1.6211525	1.1713351	0.0
0.0	    0.0	        0.0	        0.0

(1,1,4,.,.) =
0.0	    0.0	        0.0	        0.0
0.0	    0.0	        0.0	        0.0
0.0	    0.0	        0.0	        0.0
0.0	    0.0	        0.0	        0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x4x4x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import ZeroPadding3D

model = Sequential()
model.add(ZeroPadding3D((1, 1, 1), input_shape=(1, 2, 2, 2)))
input = np.random.random([1, 1, 2, 2, 2])
output = model.forward(input)
```
Input is:
```python
[[[[[0.12432462 0.19244616]
    [0.39039533 0.88140855]]

   [[0.71426182 0.86085132]
    [0.04443494 0.679125  ]]]]]
```
Output is
```python
[[[[[0.0   0.0        0.0        0.0 ]
    [0.0   0.0        0.0        0.0 ]
    [0.0   0.0        0.0        0.0 ]
    [0.0   0.0        0.0        0.0 ]]
   [[0.0   0.0        0.0        0.0 ]
    [0.0   0.12432462 0.19244616 0.0 ]
    [0.0   0.39039534 0.8814086  0.0 ]
    [0.0   0.0        0.0        0.0 ]]
   [[0.0   0.0        0.0        0.0 ]
    [0.0   0.71426183 0.8608513  0.0 ]
    [0.0   0.04443494 0.679125   0.0 ]
    [0.0   0.0        0.0        0.0 ]]
   [[0.0   0.0        0.0        0.0 ]
    [0.0   0.0        0.0        0.0 ]
    [0.0   0.0        0.0        0.0 ]
    [0.0   0.0        0.0        0.0 ]]]]]
```

---
## **UpSampling1D**
UpSampling layer for 1D inputs.

Repeats each temporal step 'length' times along the time axis.

The input of this layer should be 3D.

**Scala:**
```scala
UpSampling1D(length = 2, inputShape = null)
```
**Python:**
```python
UpSampling1D(length=2, input_shape=None, name=None)
```

**Parameters:**

* `length`: Integer. UpSampling factor. Default is 2.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, UpSampling1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(UpSampling1D(2, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.80225134	-0.9644977	 -0.71038723	-1.5673652
0.67224514	-0.24330814	 -0.082499735	0.2807591
-0.9299857	-1.8893008	 -1.1062661	    -1.1637908

(2,.,.) =
-0.1831344	-0.6621819	 -0.667329	    -0.26960346
-0.6601015	1.0819869	 1.0307902	    1.1801233
-0.18303517	0.2565441	 -0.39598823	0.23400643

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.80225134	-0.9644977	 -0.71038723	 -1.5673652
0.80225134	-0.9644977	 -0.71038723	 -1.5673652
0.67224514	-0.24330814	 -0.082499735	 0.2807591
0.67224514	-0.24330814	 -0.082499735	 0.2807591
-0.9299857	-1.8893008	 -1.1062661	    -1.1637908
-0.9299857	-1.8893008	 -1.1062661	    -1.1637908

(2,.,.) =
-0.1831344	-0.6621819	 -0.667329	    -0.26960346
-0.1831344	-0.6621819	 -0.667329	    -0.26960346
-0.6601015	1.0819869	 1.0307902	    1.1801233
-0.6601015	1.0819869	 1.0307902	    1.1801233
-0.18303517	0.2565441	 -0.39598823	0.23400643
-0.18303517	0.2565441	 -0.39598823	0.23400643

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x6x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import UpSampling1D

model = Sequential()
model.add(UpSampling1D(2, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.66227662 0.00663032 0.49010329 0.26836567]
  [0.34225774 0.26000732 0.27628499 0.49861887]
  [0.11619066 0.28123766 0.60770962 0.80773197]]

 [[0.477639   0.88906297 0.38577295 0.99058504]
  [0.50690837 0.38107999 0.05881034 0.96402145]
  [0.42226283 0.77350512 0.54961295 0.55315271]]]
```
Output is
```python
[[[0.6622766  0.00663032 0.4901033  0.26836568]
  [0.6622766  0.00663032 0.4901033  0.26836568]
  [0.34225774 0.26000732 0.276285   0.49861887]
  [0.34225774 0.26000732 0.276285   0.49861887]
  [0.11619066 0.28123766 0.60770965 0.807732  ]
  [0.11619066 0.28123766 0.60770965 0.807732  ]]

 [[0.477639   0.88906294 0.38577294 0.990585  ]
  [0.477639   0.88906294 0.38577294 0.990585  ]
  [0.50690836 0.38107997 0.05881034 0.96402144]
  [0.50690836 0.38107997 0.05881034 0.96402144]
  [0.42226282 0.7735051  0.54961294 0.5531527 ]
  [0.42226282 0.7735051  0.54961294 0.5531527 ]]]
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
import com.intel.analytics.bigdl.nn.keras.{Sequential, UpSampling2D}
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
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import UpSampling2D

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
import com.intel.analytics.bigdl.nn.keras.{Sequential, UpSampling3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(UpSampling3D((2, 2, 2), inputShape = Shape(1, 1, 2, 2)))
val input = Tensor[Float](1, 1, 1, 2, 2).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
0.9906968	-0.2451235
1.5133694	-0.34076887

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x1x2x2]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
0.9906968	0.9906968	-0.2451235	-0.2451235
0.9906968	0.9906968	-0.2451235	-0.2451235
1.5133694	1.5133694	-0.34076887	-0.34076887
1.5133694	1.5133694	-0.34076887	-0.34076887

(1,1,2,.,.) =
0.9906968	0.9906968	-0.2451235	-0.2451235
0.9906968	0.9906968	-0.2451235	-0.2451235
1.5133694	1.5133694	-0.34076887	-0.34076887
1.5133694	1.5133694	-0.34076887	-0.34076887

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x2x4x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import UpSampling3D

model = Sequential()
model.add(UpSampling3D((2, 2, 2), input_shape=(1, 1, 2, 2)))
input = np.random.random([1, 1, 1, 2, 2])
output = model.forward(input)
```
Input is:
```python
[[[[[0.58361205 0.2096227 ]
    [0.51686662 0.70260105]]]]]
```
Output is
```python
[[[[[0.583612  0.583612  0.2096227 0.2096227]
    [0.583612  0.583612  0.2096227 0.2096227]
    [0.5168666 0.5168666 0.7026011 0.7026011]
    [0.5168666 0.5168666 0.7026011 0.7026011]]

   [[0.583612  0.583612  0.2096227 0.2096227]
    [0.583612  0.583612  0.2096227 0.2096227]
    [0.5168666 0.5168666 0.7026011 0.7026011]
    [0.5168666 0.5168666 0.7026011 0.7026011]]]]]
```
