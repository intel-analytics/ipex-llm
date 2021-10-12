## **SimpleRNN**
A fully-connected recurrent neural network cell. The output is to be fed back to input.

The input of this layer should be 3D, i.e. (batch, time steps, input dim).

**Scala:**
```scala
SimpleRNN(outputDim, activation = "tanh", returnSequences = false, goBackwards = false, wRegularizer = null, uRegularizer = null, bRegularizer = null, inputShape = null)
```
**Python:**
```python
SimpleRNN(output_dim, activation="tanh", return_sequences=False, go_backwards=False, W_regularizer=None, U_regularizer=None, b_regularizer=None, input_shape=None, name=None)
```

Parameters:

* `outputDim`: Hidden unit size. Dimension of internal projections and final output.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is 'tanh'.
* `returnSequences`: Whether to return the full sequence or only return the last output in the output sequence. Default is false.
* `goBackwards`: Whether the input sequence will be processed backwards. Default is false.
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `uRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied the recurrent weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.SimpleRNN
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(SimpleRNN[Float](8, activation = "relu", inputShape = Shape(4, 5)))
val input = Tensor[Float](2, 4, 5).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.71328646	0.24269831	-0.75013286	-1.6663225	0.35494477
0.073439054	-1.1181073	-0.6577777	1.3154761	0.15396282
0.41183218	-1.2667576	-0.11167632	0.946616	0.06427766
0.013886308	-0.20620999	1.1173447	1.9083043	1.7680032

(2,.,.) =
-2.3510098	-0.8492037	0.042268332	-0.43801674	-0.010638754
1.298793	-0.24814601	0.31325665	-0.19119295	-2.072075
-0.11629801	0.27296612	0.94443846	0.37293285	-0.82289046
0.6044998	0.93386084	-1.3502276	-1.7753356	1.6173482

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x5]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.0	 0.020557694  0.0	0.39700085	0.622244  0.0	0.36524248	0.88961613
0.0	 1.4797685	  0.0	0.0	        0.0	      0.0	0.0	        0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import SimpleRNN
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(SimpleRNN(8, activation="relu", input_shape=(4, 5)))
input = np.random.random([2, 4, 5])
output = model.forward(input)
```
Input is:
```python
[[[0.43400622 0.65452575 0.94952774 0.96210478 0.05286231]
  [0.2162183  0.33225502 0.09725628 0.80813221 0.29556109]
  [0.19720487 0.35077585 0.80904872 0.80576513 0.82035253]
  [0.36175687 0.63291153 0.08437936 0.71581099 0.790709  ]]

 [[0.35387003 0.36532078 0.9834315  0.07562338 0.05600369]
  [0.65927201 0.14652252 0.10848068 0.88225065 0.88871385]
  [0.23627135 0.72620104 0.60391828 0.51571874 0.73550574]
  [0.80773506 0.35121494 0.66889362 0.530684   0.52066982]]]
```
Output is:
```python
[[0.77534926 0.23742369 0.14946866 0.0        0.16289112 0.0  0.71689016 0.24594748]
 [0.8987881  0.06123672 0.3312829  0.29757586 0.0        0.0  1.0179179  0.23447856]]
```

---
## **LSTM**
Long Short Term Memory unit architecture.

The input of this layer should be 3D, i.e. (batch, time steps, input dim).

**Scala:**
```scala
LSTM(outputDim, activation = "tanh", innerActivation = "hard_sigmoid", returnSequences = false, goBackwards = false, wRegularizer = null, uRegularizer = null, bRegularizer = null, inputShape = null)
```
**Python:**
```python
LSTM(output_dim, activation="tanh", inner_activation="hard_sigmoid", return_sequences=False, go_backwards=False, W_regularizer=None, U_regularizer=None, b_regularizer=None, input_shape=None, input_shape=None, name=None)
```

Parameters:

* `outputDim`: Hidden unit size. Dimension of internal projections and final output.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is 'tanh'.
* `innerActivation`: String representation of the activation function for inner cells. See [here](activation/#available-activations) for available activation strings. Default is 'hard_sigmoid'.
* `returnSequences`: Whether to return the full sequence or only return the last output in the output sequence. Default is false.
* `goBackwards`: Whether the input sequence will be processed backwards. Default is false.
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `uRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied the recurrent weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.
* `name`: String to set the name of the layer. If not specified, its name will by default to be a generated string.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.LSTM
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(LSTM[Float](8, inputShape = Shape(2, 3)))
val input = Tensor[Float](2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.6857518	0.21570909	-0.019308459
0.17754157	0.25172755	-1.189466

(2,.,.) =
0.23807438	1.6879119	-0.36335373
0.9826865	0.49549296	0.8100107

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.13552098	-0.043483295	-0.10553853	0.19386405	0.18295142	0.037892513	-0.05510225	-0.2420117
-0.04152686	-0.13908584	0.18151914	0.14170776	0.15598273	0.18968433	-0.042683482	-0.05782121
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import LSTM
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(LSTM(8, input_shape = (2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
array([[[ 0.67619723,  0.5168176 ,  0.8093504 ],
        [ 0.93787417,  0.53016934,  0.51934568]],

       [[ 0.57334472,  0.40007739,  0.65670337],
        [ 0.74457042,  0.15209156,  0.02015092]]])
```
Output is:
```python
array([[-0.01563799,  0.16000053, -0.20192699,  0.08859081, -0.14184587,
         0.11160418,  0.19090165,  0.03475797],
       [-0.02395577,  0.10148412, -0.13211192,  0.05772379, -0.16488783,
         0.13513438,  0.15624164,  0.02866406]], dtype=float32)
```

---
## **GRU**
Gated Recurrent Unit architecture.

The input of this layer should be 3D, i.e. (batch, time steps, input dim).

**Scala:**
```scala
GRU(outputDim, activation = "tanh", innerActivation = "hard_sigmoid", returnSequences = false, goBackwards = false, wRegularizer = null, uRegularizer = null, bRegularizer = null, inputShape = null)
```
**Python:**
```python
GRU(output_dim, activation="tanh", inner_activation="hard_sigmoid", return_sequences=False, go_backwards=False, W_regularizer=None, U_regularizer=None, b_regularizer=None, input_shape=None, name=None)
```

Parameters:

* `outputDim`: Hidden unit size. Dimension of internal projections and final output.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is 'tanh'.
* `innerActivation`: String representation of the activation function for inner cells. See [here](activation/#available-activations) for available activation strings. Default is 'hard_sigmoid'.
* `returnSequences`: Whether to return the full sequence or only return the last output in the output sequence. Default is false.
* `goBackwards`: Whether the input sequence will be processed backwards. Default is false.
* `wRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `uRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), applied the recurrent weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), applied to the bias. Default is null.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.GRU
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GRU[Float](8, inputShape = Shape(2, 3)))
val input = Tensor[Float](2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.010477358 -1.1201298  -0.86472356
0.12688802   -0.6696582  0.08027417

(2,.,.) =
0.1724209    -0.52319324 -0.8808063
0.17918338   -0.552886	 -0.11891741

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
-0.12018716	 -0.31560755	0.2867627	0.6728765	0.13287778	0.2112865	0.13381396	-0.4267934
-0.18521798	 -0.30512968	0.14875418	0.63962734	0.1841841	0.25272882	0.016909363	-0.38463163
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import GRU
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(GRU(8, input_shape=(2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.25026651 0.35433442 0.01417391]
  [0.77236921 0.97315472 0.66090386]]

 [[0.76037554 0.41029034 0.68725938]
  [0.17888889 0.67670088 0.70580547]]]
```
Output is:
```python
[[-0.03584666  0.07984452 -0.06159414 -0.13331707  0.34015405 -0.07107028  0.12444386 -0.06606203]
 [ 0.02881907  0.04856917 -0.15306929 -0.24991018  0.23814955  0.0303434   0.06634206 -0.15335503]]
```

---
## **Highway**
Densely connected highway network.

Highway layers are a natural extension of LSTMs to feedforward networks.

The input of this layer should be 2D, i.e. (batch, input dim).

**Scala:**
```scala
Highway(activation = null, wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
Highway(activation=None, W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

Parameters:

* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `wRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.Highway
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Highway[Float](inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.26041138	0.4286919	1.723103
1.4516269	0.5557163	-0.1149741
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
-0.006746907	-0.109112576	1.3375516
0.6065166	0.41575465	-0.06849813
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Highway
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Highway(input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.5762107  0.45679288 0.00370956]
 [0.24133312 0.38104653 0.05249192]]
```
Output is:
```python
[[0.5762107  0.4567929  0.00370956]
 [0.24133313 0.38104653 0.05249191]]
```
