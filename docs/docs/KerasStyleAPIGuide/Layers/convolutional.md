## **LocallyConnected2D**
A Locally-connected layer for 2D input works similarly to a SpatialConvolution layer, except that weights are unshared, that is, a different set of filters is applied at different patch of the input.

The input is 2D tensor with shape: (batch_size, channels, rows, cols).

**Scala:**
```scala
LocallyConnected2D(nbFilter, nbRow, nbCol, activation = null, borderMode = "valid", subsample = (1, 1), dimOrdering = "th", wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
LocallyConnected2D(nb_filter, nb_row, nb_col, activation=None, border_mode="valid", subsample=(1, 1), dim_ordering="th", W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

**Parameters:**

* `nbFilter`: Number of convolution filters.
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
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.LocallyConnected2D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(LocallyConnected2D[Float](2, 2, 2, inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-0.71993834     0.018790463     0.08133635      0.35603827
-1.1757486      1.8503827       -1.4548069      -0.6309117
-0.53039306     -0.14174776     0.7653523       -0.1891388

(1,2,.,.) =
1.0949191       0.13689162      0.35839355      -0.14805469
-2.5264592      -0.34186792     1.3190275       -0.11725446
-0.48823252     -1.5305915      -1.0556486      1.792275

(2,1,.,.) =
0.92393816      0.83243525      0.22506136      0.6694662
0.7662836       -0.23876576     -0.7719174      0.13114463
0.042082224     1.2212821       -1.2496184      -0.18717249

(2,2,.,.) =
0.726698        0.42673108      0.0786712       -1.4069401
-0.090565465    0.49527475      0.08590904      -0.51858175
1.4575573       0.9669369       0.21832618      0.34654656

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.022375792     0.669761        -0.25723624
0.99919814      0.93189466      0.8592935

(1,2,.,.) =
0.12613812      -1.0531536      0.8148589
0.66276294      0.12609969      0.6590149

(2,1,.,.) =
-0.1259023      0.32203823      0.07248953
-0.125191       -0.1285046      0.021367729

(2,2,.,.) =
-0.13560611     -0.038621478    -0.08420516
-0.0021556932   -0.094522506    -0.08551059

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import LocallyConnected2D

model = Sequential()
model.add(LocallyConnected2D(2, 2, 2, input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.75179142 0.10678918 0.92663152 0.2041142 ]
   [0.03534582 0.13742629 0.94115987 0.17303432]
   [0.91112368 0.19837546 0.45643767 0.16589123]]

  [[0.22996923 0.22878544 0.75623624 0.7058976 ]
   [0.14107232 0.49484648 0.71194356 0.53604538]
   [0.46257205 0.46902871 0.48046811 0.83579709]]]


 [[[0.9397535  0.51814825 0.10492714 0.24623405]
   [0.69800376 0.12353963 0.69536497 0.05159074]
   [0.56722731 0.33348394 0.47648031 0.25398067]]

  [[0.51018599 0.3416568  0.14112375 0.76505795]
   [0.16242231 0.16735028 0.79000471 0.98701885]
   [0.79852431 0.77458166 0.12551857 0.43866238]]]]
```
Output is
```python
[[[[ 0.14901309 -0.11168094  0.28349853]
   [ 0.21792562  0.49922782 -0.06560349]]

  [[ 0.6176302  -0.4638375  -0.13387583]
   [-0.04903107  0.07764787 -0.33653474]]]


 [[[ 0.24676235 -0.46874076  0.33973938]
   [ 0.21408634  0.36619198  0.17972258]]

  [[ 0.35941058 -0.23446569 -0.09271184]
   [ 0.39490524 -0.00668371 -0.25355732]]]]
```

---
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
## **Convolution2D**
Applies a 2D convolution over an input image composed of several input planes.

You can also use `Conv2D` as an alias of this layer.

The input of this layer should be 4D, i.e. (samples, channels, rows, cols).
The output of this layer should be 4D, i.e. (samples, filters, new_rows, new_cols).

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
* `name`: String to set the name of the layer. If not specified, its name will by default to be a generated string.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.Convolution2D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Convolution2D[Float](4, 2, 2, activation = "relu", inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.8852683 -0.81495345 -1.2799169  0.9779215
1.1456866 -0.10803124 -0.44350016 -1.7670554
-0.9059258  -0.08115104 -0.888267 1.8203543

(1,2,.,.) =
-0.69458634 0.31331652  1.4600077 -0.93392456
1.4808512 0.2082488 -0.008410408  0.013914147
0.86024827  1.124567  0.28874534  -0.4866409

(2,1,.,.) =
-0.020653103  0.8077344 -0.9391865  0.2743323
0.09707443  -0.1877453  2.3798819 1.71017
0.14860597  0.8954743 2.0009918 1.0548053

(2,2,.,.) =
-0.06750481 -2.1010966  -0.51831937 -0.40519416
1.2983296 1.9960507 0.31097296  -1.0400984
-0.20703147 0.32478333  -0.5247251  1.2356688

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.49652004  0.62284863  0.0
1.2256577 0.11462581  0.761484

(1,2,.,.) =
0.0 0.0 1.6321466
0.69082737  0.10713227  0.0

(1,3,.,.) =
0.0 0.0 1.0226117
0.0 0.0 0.0

(1,4,.,.) =
0.017812707 0.044630717 0.0
0.0 0.0 0.0

(2,1,.,.) =
0.0 0.79017955  0.0
1.1551664 0.0 0.0

(2,2,.,.) =
0.0 0.0 0.0
0.0 0.9762883 0.0

(2,3,.,.) =
0.0 0.0 0.0
0.0 0.0 0.0

(2,4,.,.) =
0.0 0.0 0.1633394
0.66279346  0.07180607  1.7188346

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Convolution2D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Convolution2D(4, 2, 2, input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
array([[[[ 0.70766604,  0.56604946,  0.89172683,  0.35057259],
         [ 0.89700606,  0.71675588,  0.92357667,  0.73319623],
         [ 0.38198447,  0.66954234,  0.46397678,  0.81329758]],

        [[ 0.86972625,  0.16386155,  0.73140259,  0.07359015],
         [ 0.43441431,  0.16852341,  0.15025034,  0.34109183],
         [ 0.89670592,  0.06335869,  0.72356566,  0.54245763]]],


       [[[ 0.37727322,  0.14688331,  0.06249512,  0.29553298],
         [ 0.50554043,  0.33364744,  0.95334248,  0.40551935],
         [ 0.81317402,  0.59253283,  0.8249684 ,  0.80419637]],

        [[ 0.71737738,  0.09376579,  0.3793706 ,  0.91432729],
         [ 0.34433954,  0.74886398,  0.97859311,  0.9538775 ],
         [ 0.45521369,  0.79446047,  0.35239537,  0.12803574]]]])
```
Output is
```python
array([[[[ 0.0732559 ,  0.70261478,  0.16962567],
         [ 0.3641817 ,  0.56304729,  0.71597064]],

        [[-0.5932048 , -0.04155506, -0.49025974],
         [-0.57992101, -0.00230447, -0.33811107]],

        [[ 0.13634545,  0.27157408, -0.01450583],
         [ 0.34469086,  0.46334854,  0.55308509]],

        [[-0.01247289,  0.69034004, -0.01554111],
         [ 0.07790593,  0.09984782,  0.1278697 ]]],


       [[[ 0.02547407,  0.64045584,  0.21886043],
         [ 0.43482357,  0.45493811,  0.26216859]],

        [[-0.39469361, -0.34455007, -0.2396858 ],
         [-0.15447566, -0.35714447, -0.44134659]],

        [[ 0.30956799,  0.9154281 ,  0.75450832],
         [ 0.37207305,  0.55432665, -0.29964659]],

        [[-0.48307419, -0.29406634, -0.29416537],
         [ 0.0138942 ,  0.26592475,  0.38921899]]]], dtype=float32)
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
(1,1,.,.) 
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
Output is:
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
    [0.01897243 0.01897243 0.87927955 0.87927955]
    [0.13656585 0.13656585 0.3003842  0.3003842 ]
    [0.13656585 0.13656585 0.3003842  0.3003842 ]]

   [[0.01897243 0.01897243 0.87927955 0.87927955]
    [0.01897243 0.01897243 0.87927955 0.87927955]
    [0.13656585 0.13656585 0.3003842  0.3003842 ]
    [0.13656585 0.13656585 0.3003842  0.3003842 ]]]]]
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
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

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

---
## **Cropping1D**
Cropping layer for 1D input (e.g. temporal sequence).

It crops along the time dimension (axis 1). 

The input of this layer should be 3D, i.e. (batch, axis_to_crop, features).
The output of this layer should be 3D, i.e. (batch, cropped_axis, features).

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
* `name`: String to set the name of the layer. If not specified, its name will by default to be a generated string.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.Cropping1D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Cropping1D[Float]((1, 1), inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.06297628	-0.8408224	0.21813048	-0.14371997
0.9278932	0.069493145	-0.2900171	0.536517
3.430168	-0.53643423	0.12677099	0.3572487

(2,.,.) =
1.493348	-1.1703341	-0.37385875	-0.239736
0.33984247	-0.6005885	1.2722077	-0.5043763
0.012092848	0.40293974	0.61356264	2.4283617

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.9278932	0.069493145	-0.2900171	0.536517

(2,.,.) =
0.33984247	-0.6005885	1.2722077	-0.5043763

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1x4]
```

**Python example:**
```python
from zoo.pipeline.api.keras.layers import Cropping1D
from zoo.pipeline.api.keras.models import Sequential
import numpy as np

model = Sequential()
model.add(Cropping1D(input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
array([[[ 0.12013423,  0.21359734,  0.92871231,  0.92152503],
        [ 0.3649771 ,  0.39968689,  0.92007275,  0.16493056],
        [ 0.11018303,  0.7591447 ,  0.35932136,  0.97727728]],

       [[ 0.06645696,  0.21909036,  0.01219254,  0.46561466],
        [ 0.64316144,  0.53577975,  0.38302965,  0.56807556],
        [ 0.25223652,  0.23857826,  0.1884081 ,  0.42532243]]])
```
Output is:
```python
array([[[ 0.36497709,  0.3996869 ,  0.92007273,  0.16493057]],

       [[ 0.64316142,  0.53577977,  0.38302964,  0.56807554]]], dtype=float32)
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
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.Cropping2D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Cropping2D[Float](((0, 1), (1, 0)), inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-0.6840084      0.293568        0.045959193     0.91535753
-0.49666363     -0.05026308     0.22163485      0.08330725
0.36190453      -0.023894459    0.40037137      0.15155333

(1,2,.,.) =
1.0107938       0.05100493      -0.88689697     0.111396775
0.065911256     -0.41727677     0.62742686      -0.5435138
-1.0133605      0.7352207       -0.77922934     -0.36588958

(2,1,.,.) =
-0.6847248      0.8627568       -0.5600547      0.48514402
-0.9261762      -0.34248486     -0.09243064     -0.13134436
-0.23247129     1.2801572       -1.377833       -1.7608607

(2,2,.,.) =
1.1907105       0.30009162      -1.2604285      1.0099201
-1.211673       -0.08809458     0.4386406       -0.6264226
0.112140626     0.3690179       0.832656        1.3931179
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.293568        0.045959193     0.91535753
-0.05026308     0.22163485      0.08330725

(1,2,.,.) =
0.05100493      -0.88689697     0.111396775
-0.41727677     0.62742686      -0.5435138
(2,1,.,.) =
0.8627568       -0.5600547      0.48514402
-0.34248486     -0.09243064     -0.13134436
(2,2,.,.) =
0.30009162      -1.2604285      1.0099201
-0.08809458     0.4386406       -0.6264226

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x3]
```

**Python example:**
```python
from zoo.pipeline.api.keras.layers import Cropping2D
from zoo.pipeline.api.keras.models import Sequential
import numpy as np

model = Sequential()
model.add(Cropping2D(((0, 1), (1, 0)), input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
array([[[[0.04386121, 0.78710294, 0.4518868 , 0.78738097],
         [0.36859968, 0.44601991, 0.94679033, 0.93842937],
         [0.55705904, 0.30684226, 0.90630488, 0.9323689 ]],

        [[0.32265899, 0.37304445, 0.09097587, 0.52496901],
         [0.70275446, 0.10796127, 0.74849378, 0.99118752],
         [0.34310691, 0.60435919, 0.22227177, 0.48464358]]],
       [[[0.93479186, 0.6009071 , 0.09771059, 0.19654216],
         [0.48278365, 0.0968289 , 0.9465143 , 0.49814986],
         [0.36140084, 0.98581155, 0.14834531, 0.71290525]],

        [[0.8909849 , 0.66729728, 0.53332039, 0.83958965],
         [0.3645429 , 0.40645471, 0.02596942, 0.80835778],
         [0.62524417, 0.14305505, 0.6706279 , 0.4283277 ]]]])
```
Output is:
```python
array([[[[0.78710294, 0.4518868 , 0.787381  ],
         [0.44601992, 0.94679034, 0.93842936]],

        [[0.37304446, 0.09097587, 0.524969  ],
         [0.10796127, 0.7484938 , 0.9911875 ]]],
       [[[0.6009071 , 0.09771059, 0.19654216],
         [0.0968289 , 0.9465143 , 0.49814987]],

        [[0.6672973 , 0.53332037, 0.83958966],
         [0.4064547 , 0.02596942, 0.8083578 ]]]], dtype=float32)
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
* `dimOrdering`: Format of input data. Either 'CHANNEL_FIRST' (dimOrdering='th') or 'CHANNEL_LAST' (dimOrdering='tf'). Default is 'CHANNEL_FIRST'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.Cropping3D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Cropping3D[Float](((1, 1), (1, 1), (1, 1)), inputShape = Shape(2, 3, 4, 5)))
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
from zoo.pipeline.api.keras.layers import Cropping3D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Cropping3D(((1, 1), (1, 1), (1, 1)), input_shape=(2, 3, 4, 5)))
input = np.random.random([2, 2, 3, 4, 5])
output = model.forward(input)
```
Input is:
```python
array([[[[[0.62840716, 0.49718584, 0.12585459, 0.45339446, 0.51496759],
          [0.09154417, 0.31975017, 0.45159785, 0.69461629, 0.01777911],
          [0.03056908, 0.58578471, 0.4212357 , 0.81290609, 0.54614353],
          [0.56553699, 0.42969119, 0.55706099, 0.57701881, 0.41386126]],

         [[0.84399973, 0.79438576, 0.72216539, 0.24147284, 0.02302575],
          [0.88659717, 0.65307522, 0.47795438, 0.18358642, 0.10409304],
          [0.02787308, 0.57958405, 0.78614037, 0.12632357, 0.96611954],
          [0.03602844, 0.29878791, 0.59278562, 0.25408987, 0.60823159]],

         [[0.07057682, 0.8308839 , 0.27391967, 0.90192561, 0.80467445],
          [0.50686651, 0.6975992 , 0.89386305, 0.33915142, 0.30557542],
          [0.58812313, 0.41667892, 0.0859111 , 0.21376582, 0.06077911],
          [0.3321846 , 0.77915362, 0.80878924, 0.44581895, 0.87659508]]],

        [[[0.42478273, 0.41505405, 0.86690148, 0.81330225, 0.85384093],
          [0.9370089 , 0.18919117, 0.92571803, 0.82038262, 0.75380295],
          [0.48092604, 0.27035346, 0.30137481, 0.33337198, 0.88508334],
          [0.44941603, 0.59172234, 0.02723888, 0.3714394 , 0.63989379]],

         [[0.39549828, 0.19292932, 0.91677619, 0.40739894, 0.63731699],
          [0.91693476, 0.89300681, 0.8599061 , 0.38889494, 0.55620744],
          [0.8269569 , 0.45751382, 0.1316247 , 0.04326183, 0.71251854],
          [0.56835414, 0.75783607, 0.6697517 , 0.55425787, 0.1779235 ]],

         [[0.97761621, 0.12224875, 0.0565609 , 0.88227811, 0.15135005],
          [0.9700492 , 0.590918  , 0.88279087, 0.36807701, 0.48872168],
          [0.847832  , 0.64009568, 0.97971251, 0.06989564, 0.80387185],
          [0.33721551, 0.99582496, 0.4309207 , 0.77468415, 0.17438985]]]],

       [[[[0.52570481, 0.15825837, 0.96653256, 0.8395669 , 0.33314475],
          [0.44051007, 0.66105309, 0.44270763, 0.46340145, 0.09020919],
          [0.4220039 , 0.75622627, 0.66531762, 0.5474585 , 0.95511606],
          [0.8150854 , 0.12041384, 0.16459857, 0.90216744, 0.90415106]],

         [[0.23274933, 0.78995579, 0.8205956 , 0.0098613 , 0.39972397],
          [0.46246117, 0.68833063, 0.76978062, 0.14479477, 0.80658274],
          [0.29013113, 0.03855975, 0.12752528, 0.97587177, 0.22943272],
          [0.61845944, 0.39336312, 0.70661959, 0.58377891, 0.41844674]],

         [[0.04968886, 0.83604265, 0.82907304, 0.05302717, 0.15273231],
          [0.5287088 , 0.54298116, 0.46370681, 0.23882016, 0.93293435],
          [0.44967435, 0.44840028, 0.46009438, 0.68473051, 0.26375504],
          [0.04099288, 0.4334504 , 0.08448742, 0.92742616, 0.21594092]]],

        [[[0.99377422, 0.10287153, 0.95161776, 0.41423906, 0.2863645 ],
          [0.30002606, 0.43550723, 0.87747421, 0.41472721, 0.91166764],
          [0.41821649, 0.84575542, 0.92085315, 0.85144318, 0.45106024],
          [0.12081268, 0.86000088, 0.61870455, 0.16207645, 0.96441056]],

         [[0.67447583, 0.07718448, 0.45813553, 0.38294045, 0.47993   ],
          [0.60947025, 0.66391439, 0.49371347, 0.92276753, 0.5735208 ],
          [0.19690983, 0.58194273, 0.8964776 , 0.51749435, 0.13312089],
          [0.88902345, 0.92261557, 0.00146803, 0.76453644, 0.91164938]],

         [[0.15939257, 0.14745922, 0.75721476, 0.44560904, 0.30039002],
          [0.80775365, 0.96551208, 0.95964112, 0.94420177, 0.42949841],
          [0.26737604, 0.81199024, 0.05778487, 0.15004785, 0.55616372],
          [0.51186541, 0.96281586, 0.36559551, 0.79961066, 0.69312035]]]]])
```
Output is:
```python
array([[[[[0.6530752 , 0.4779544 , 0.18358642],
          [0.57958406, 0.7861404 , 0.12632357]]],

        [[[0.8930068 , 0.8599061 , 0.38889495],
          [0.4575138 , 0.1316247 , 0.04326183]]]],
       [[[[0.68833065, 0.76978064, 0.14479478],
          [0.03855975, 0.12752528, 0.9758718 ]]],

        [[[0.6639144 , 0.49371347, 0.9227675 ],
          [0.58194274, 0.8964776 , 0.5174943 ]]]]], dtype=float32)
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
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.ZeroPadding2D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(ZeroPadding2D[Float]((1, 1), inputShape = Shape(2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
1.2227936       0.30803198      -1.3921114
0.43359384      -0.038079295    -1.241585

(1,2,.,.) =
-1.1766883      -2.015887       -0.7110933
-0.5415997      -0.50294536     -1.3715594
(2,1,.,.) =
0.10733734      1.3369694       0.037685163
-1.2942516      0.2693859       0.6846867
(2,2,.,.) =
-1.4678168      0.21972063      0.40070927
0.45242524      -0.03342953     -0.8016073

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.0     0.0     0.0     0.0     0.0
0.0     1.2227936       0.30803198      -1.3921114      0.0
0.0     0.43359384      -0.038079295    -1.241585       0.0
0.0     0.0     0.0     0.0     0.0

(1,2,.,.) =
0.0     0.0     0.0     0.0     0.0
0.0     -1.1766883      -2.015887       -0.7110933      0.0
0.0     -0.5415997      -0.50294536     -1.3715594      0.0
0.0     0.0     0.0     0.0     0.0

(2,1,.,.) =
0.0     0.0     0.0     0.0     0.0
0.0     0.10733734      1.3369694       0.037685163     0.0
0.0     -1.2942516      0.2693859       0.6846867       0.0
0.0     0.0     0.0     0.0     0.0

(2,2,.,.) =
0.0     0.0     0.0     0.0     0.0
0.0     -1.4678168      0.21972063      0.40070927      0.0
0.0     0.45242524      -0.03342953     -0.8016073      0.0
0.0     0.0     0.0     0.0     0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x4x5]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import ZeroPadding2D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(ZeroPadding2D(input_shape=(2, 2, 3)))
input = np.random.random([2, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
array([[[[0.0544422 , 0.21723616, 0.69071413],
         [0.68166784, 0.78673863, 0.63838101]],

        [[0.43930351, 0.62153019, 0.5539688 ],
         [0.79930636, 0.07007638, 0.13261168]]],
       [[[0.21493318, 0.21060602, 0.12101637],
         [0.90132665, 0.95799647, 0.09733214]],

        [[0.21548934, 0.27369217, 0.06024094],
         [0.85388521, 0.63911987, 0.34428558]]]])
```
Output is:
```python
array([[[[0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.0544422 , 0.21723616, 0.6907141 , 0.        ],
         [0.        , 0.68166786, 0.78673863, 0.638381  , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        ]],

        [[0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.43930352, 0.6215302 , 0.5539688 , 0.        ],
         [0.        , 0.79930633, 0.07007638, 0.13261168, 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        ]]],

       [[[0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.21493319, 0.21060602, 0.12101637, 0.        ],
         [0.        , 0.90132666, 0.9579965 , 0.09733213, 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        ]],

        [[0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.21548934, 0.27369216, 0.06024094, 0.        ],
         [0.        , 0.85388523, 0.63911986, 0.34428558, 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        ]]]],
      dtype=float32)
```

---
## **ShareConvolution2D**
Applies a 2D convolution over an input image composed of several input planes.

You can also use ShareConv2D as an alias of this layer.

Data format currently supported for this layer is DataFormat.NCHW (dimOrdering='th').

The input of this layer should be 4D.

**Scala:**
```scala
ShareConvolution2D(nbFilter, nbRow, nbCol, init = "glorot_uniform", activation = null, subsample = (1, 1), padH = 0, padW = 0, 
                   propagateBack = true, dimOrdering = "th", wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
ShareConvolution2D(nb_filter, nb_row, nb_col, init="glorot_uniform", activation=None, subsample=(1, 1), pad_h=0, pad_w=0,
                   propagate_back=True, dim_ordering="th", W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

**Parameters:**

* `nbFilter`: Number of convolution filters to use.
* `nbRow`: Number of rows in the convolution kernel.
* `nbCol`: Number of columns in the convolution kernel.
* `init`: Initialization method for the weights of the layer. Default is Xavier.
          You can also pass in corresponding string representations such as 'glorot_uniform'
          or 'normal', etc. for simple init methods in the factory method.
* `activation`: Activation function to use. Default is null.
                You can also pass in corresponding string representations such as 'relu'
                or 'sigmoid', etc. for simple activations in the factory method.
* `subsample`: Int array of length 2 corresponding to the step of the convolution in the
               height and width dimension. Also called strides elsewhere. Default is (1, 1).
* `padH`: The additional zeros added to the height dimension. Default is 0.
* `padW`: The additional zeros added to the width dimension. Default is 0.
* `propagateBack`: Whether to propagate gradient back. Default is true.
* `dimOrdering`: Format of input data. Please use DataFormat.NCHW (dimOrdering='th').
* `wRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), (eg. L1 or L2 regularization),
                  applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.ShareConvolution2D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(ShareConvolution2D[Float](nbFilter = 2, nbRow = 2, nbCol = 3, inputShape = Shape(2, 2, 3)))
val input = Tensor[Float](1, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.033261865     -0.5991786      1.7385886
-0.56382173     0.4827164       -0.62269926

(1,2,.,.) =
-0.31000894     -0.05032834     -1.1754748
2.594314        -1.0447274      -1.2348005

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
-0.39924833

(1,2,.,.) =
-0.05582048

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x1x1]
```

**Python example:**
```python
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import ShareConvolution2D
import numpy as np

model = Sequential()
model.add(ShareConvolution2D(2, 2, 3, input_shape=(2, 2, 3)))
input = np.random.random([1, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
array([[[[0.94476901, 0.20822355, 0.12900894],
         [0.07171242, 0.40400603, 0.87892258]],

        [[0.40369527, 0.92786425, 0.17116734],
         [0.73204729, 0.89770083, 0.86390069]]]])
```
Output is
```python
array([[[[ 0.1860767 ]],

        [[-0.00958405]]]], dtype=float32)
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
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.UpSampling1D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(UpSampling1D[Float](length = 3, inputShape = Shape(2, 3)))
val input = Tensor[Float](2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.8499613      0.6955453       -2.8545783
-0.26392975     -0.5695636      0.13427743

(2,.,.) =
0.52427506      -0.7843101      -0.12673262
1.0643414       0.69714475      -0.013671399

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.8499613      0.6955453       -2.8545783
-0.8499613      0.6955453       -2.8545783
-0.8499613      0.6955453       -2.8545783
-0.26392975     -0.5695636      0.13427743
-0.26392975     -0.5695636      0.13427743
-0.26392975     -0.5695636      0.13427743

(2,.,.) =
0.52427506      -0.7843101      -0.12673262
0.52427506      -0.7843101      -0.12673262
0.52427506      -0.7843101      -0.12673262
1.0643414       0.69714475      -0.013671399
1.0643414       0.69714475      -0.013671399
1.0643414       0.69714475      -0.013671399

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x6x3]
```

**Python example:**
```python
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import UpSampling1D
import numpy as np

model = Sequential()
model.add(UpSampling1D(length=3, input_shape=(2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
array([[[0.22908319, 0.6684591 , 0.12425427],
        [0.02378978, 0.12953109, 0.70786959]],

       [[0.40711686, 0.64417535, 0.92019981],
        [0.28788481, 0.77902591, 0.93019748]]])
```
Output is
```python
array([[[0.2290832 , 0.6684591 , 0.12425426],
        [0.2290832 , 0.6684591 , 0.12425426],
        [0.2290832 , 0.6684591 , 0.12425426],
        [0.02378978, 0.12953109, 0.7078696 ],
        [0.02378978, 0.12953109, 0.7078696 ],
        [0.02378978, 0.12953109, 0.7078696 ]],

       [[0.40711686, 0.64417535, 0.9201998 ],
        [0.40711686, 0.64417535, 0.9201998 ],
        [0.40711686, 0.64417535, 0.9201998 ],
        [0.2878848 , 0.7790259 , 0.9301975 ],
        [0.2878848 , 0.7790259 , 0.9301975 ],
        [0.2878848 , 0.7790259 , 0.9301975 ]]], dtype=float32)
```