## **Conv1D**
1D convolution layer (e.g. temporal convolution).

This layer creates a convolution kernel that is convolved
with the layer input over a single spatial (or temporal) dimension
to produce a tensor of outputs.
If `use_bias` is True, a bias vector is created and added to the outputs.
Finally, if `activation` is not `None`,
it is applied to the outputs as well.

When using this layer as the first layer in a model,
provide an `input_shape` argument
(tuple of integers or `None`, e.g.
`(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

Input shape:
        
3D tensor with shape: `(batch_size, steps, input_dim)`

Output shape:

3D tensor with shape: `(batch_size, new_steps, filters)`
`steps` value might have changed due to padding or strides.

**Scala:**
```scala
Conv1D(filters, kernelSize, strides=1, padding = "valid", activation = null, useBias = True, kernelInitializer = "glorot_uniform", biasInitializer = "zero", kernelRegularizer = null, biasRegularizer = null, inputShape = null)
```
**Python:**
```python
Conv1D(filters, kernel_size, strides=1, padding="valid", activation=None, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer = "zero", kernel_regularizer=None, bias_regularizer=None, input_shape=None, name=None)
```

**Parameters:**

* `filters`: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
* `kernel_size`: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
* `strides`: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
* `padding`: One of `"valid"` or `"same"` (case-insensitive).
            `"valid"` means "no padding".
            `"same"` results in padding the input such that
            the output has the same length as the original input.
* `activation`: Activation function to use. If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
* `use_bias`: Boolean, whether the layer uses a bias vector.
* `kernel_initializer`: Initializer for the `kernel` weights matrix. Default is 'glorot_uniform'.
* `bias_initializer`: Initializer for the bias vector.
* `kernel_regularizer`: Regularizer function applied to
            the `kernel` weights matrix. Default is null.
* `bias_regularizer`: Regularizer function applied to the bias vector. Default is null.
* `input_shape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a shape object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Conv1D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Conv1D[Float](8, 3, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.3490186	0.9116212	1.0265731	-1.1517781	
-1.3408363	0.068315335	2.330641	0.15831113	
-0.3477347	-0.31537533	-0.004820011	0.19639632	

(2,.,.) =
-2.5452073	-0.07062272	0.07531657	0.7308297	
-0.5541283	-0.672619	0.4120175	-0.63392377	
-1.7937882	1.178323	-0.9584365	0.35273483

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: 
(1,.,.) =
1.9344429	-0.67874485	-2.8546433	-0.6660415	2.754292	0.91595435	-0.32557443	0.25574505	

(2,.,.) =
-0.15532638	-1.7119135	-0.53497326	-1.4706889	0.4749836	2.0963004	-0.32759145	-2.57343	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras2.layers import Conv1D

model = Sequential()
model.add(Conv1D(8, 3, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.26717132 0.71646316 0.46861815 0.15532447]
  [0.43466464 0.48129897 0.31993028 0.17653698]
  [0.49705025 0.36898188 0.17595273 0.13695961]]

 [[0.05537173 0.62180016 0.36560319 0.95572837]
  [0.98196495 0.20136646 0.0423306  0.02030028]
  [0.65687877 0.91620089 0.37612963 0.52101501]]]
```
Output is
```python
[[[ 0.14253668  0.4923424   0.08080368 -0.5239319  -0.26981926  0.57145274  0.48187608  1.4999257 ]]

 [[-0.10191379  0.51799184 -0.20100947 -0.7895199  -0.23697199  0.4781017   0.5892592   2.1963782 ]]]
```

---
## **Conv2D**
2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of
outputs. If `use_bias` is True,
a bias vector is created and added to the outputs. Finally, if
`activation` is not `None`, it is applied to the outputs as well.

When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers, does not include the sample axis),
e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
in `data_format="channels_last"`.

Input shape:

4D tensor with shape:
`(samples, channels, rows, cols)` if data_format='channels_first'
or 4D tensor with shape:
`(samples, rows, cols, channels)` if data_format='channels_last'.

Output shape:

4D tensor with shape:
`(samples, filters, new_rows, new_cols)` if data_format='channels_first'
or 4D tensor with shape:
`(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
`rows` and `cols` values might have changed due to padding.

**Scala:**
```scala
Conv2D(filters, kernelSize, strides, padding = "valid", dataFormat = "channels_first", activation = null, useBias = true, kernelInitializer = "glorot_uniform", biasInitializer = "zero", kernelRegularizer = null, biasRegularizer = null, inputShape = null)
```
**Python:**
```python
Conv2D(filters, kernel_size, strides=(1, 1), padding="valid", data_format="channels_first", activation=None, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zero", kernel_regularizer=None, bias_regularizer=None, input_shape=None, name=None)
```

**Parameters:**

* `filters`: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
* `kernel_size`: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
* `strides`: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".            
* `padding`: one of `"valid"` or `"same"` (case-insensitive).
* `data_format`: Number of columns in the convolution kernel.
* `activation`: Activation function to use. 
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
* `use_bias`: Boolean, whether the layer uses a bias vector. Default is true.
* `kernel_initializer`: Initializer for the `kernel` weights matrix. Default is 'glorot_uniform'.
* `bias_initializer`: Initializer for the bias vector.
* `kernel_regularizer`: Regularizer function applied to
            the `kernel` weights matrix. Default is null.
* `bias_regularizer`: Regularizer function applied to the bias vector. Default is null.
* `input_shape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a shape object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Conv2D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Conv2D[Float](4, Array(2, 2), activation = "relu", inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-1.7008811	0.12288668	1.0145894	0.34869674	
0.1495685	-0.84290063	-0.13881613	0.22891551	
-0.065949395	0.7776933	0.3898055	0.8187307	

(1,2,.,.) =
-0.6645274	0.44756833	-0.8268243	-0.13796453	
-1.2200167	-0.89445364	-0.16754562	-0.7029418	
-0.032706447	-1.2504591	0.24031237	0.8331628	

(2,1,.,.) =
0.025527362	-1.456607	1.4085853	2.115896	
0.28405094	2.473169	-2.1256483	-0.37065008	
1.1322745	2.3098936	0.40274113	-0.009792422	

(2,2,.,.) =
0.38416716	0.42884415	0.48050597	-0.32054836	
-0.18368138	-0.83845645	0.69398314	0.81973153	
0.30809402	1.1508962	0.8602869	-0.27299604	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: 
(1,1,.,.) =
0.0	0.0	0.51253283	
0.09132275	0.0	0.0	

(1,2,.,.) =
0.96514046	0.32586247	0.0	
0.0	0.7704669	0.0	

(1,3,.,.) =
0.41404063	0.35680038	0.15091634	
0.0	0.0	0.0	

(1,4,.,.) =
0.0	0.5466105	0.0	
0.0	0.0	0.45137247	

(2,1,.,.) =
0.0	0.88202	0.0	
0.0	2.1273396	0.0	

(2,2,.,.) =
0.0	0.90696275	0.0	
0.0	0.0	0.28731248	

(2,3,.,.) =
0.0	0.0	0.5000323	
0.0	0.0	0.0	

(2,4,.,.) =
0.3865677	0.0	0.8942361	
1.1031605	0.0	1.0433162	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras2.layers import Conv2D

model = Sequential()
model.add(Conv2D(4, (2, 2), input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.80687428 0.17910722 0.42103319 0.62854611]
   [0.15771997 0.63882001 0.47435052 0.74072266]
   [0.75874837 0.89953139 0.98486427 0.78619311]]

  [[0.17276155 0.6644038  0.06076521 0.93869801]
   [0.12812916 0.47968789 0.87123    0.15238281]
   [0.54624731 0.48725399 0.65683408 0.40533143]]]


 [[[0.67267837 0.08983171 0.77597291 0.64776813]
   [0.23830409 0.80719787 0.50198151 0.87555294]
   [0.83633492 0.84309489 0.54959086 0.09094626]]

  [[0.37056036 0.61459394 0.79002544 0.74196783]
   [0.33012708 0.90385893 0.45793861 0.89058154]
   [0.72228852 0.71115986 0.1502346  0.85841747]]]]
```
Output is
```python
[[[[ 0.18198788 -0.07777633 -0.06771126]
   [ 0.11002313  0.09865391  0.13217448]]

  [[-0.41738698  0.0429197  -0.53935146]
   [-0.3684827  -0.61656874 -0.40491766]]

  [[ 0.15622243  0.31875694  0.4548544 ]
   [ 0.46874833  0.3621596   0.73849726]]

  [[ 0.21264602  0.73464984  0.38783965]
   [ 0.44780225  0.69555914  0.7172658 ]]]


 [[[-0.02473125 -0.26770562  0.06244571]
   [-0.04517556  0.03924857  0.12935217]]

  [[-0.21954477 -0.37033984 -0.44978783]
   [-0.45664912 -0.5516143  -0.34895533]]

  [[ 0.16499116  0.45316857  0.51688266]
   [ 0.4307469   0.5550571   0.06971958]]

  [[ 0.54072773  0.8459399   0.7282518 ]
   [ 0.71461993  0.6869658   0.5307025 ]]]]
```
