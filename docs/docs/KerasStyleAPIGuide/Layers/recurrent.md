---
## **SimpleRNN**
A fully-connected recurrent neural network cell. The output is to be fed back to input.

The input of this layer should be 3D, i.e. (batch, time steps, input dim).

**Scala:**
```scala
SimpleRNN(outputDim, activation, returnSequences = false, goBackwards = false, wRegularizer = null, uRegularizer = null, bRegularizer = null, inputShape = null)
```
**Python:**
```python
SimpleRNN(output_dim, activation="tanh", return_sequences=False, go_backwards=False, W_regularizer=None, U_regularizer=None, b_regularizer=None, input_shape=None)
```

Parameters:

* `outputDim`: Hidden unit size. Dimension of internal projections and final output.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is 'tanh'.
* `returnSequences`: Whether to return the full sequence or only return the last output in the output sequence. Default is false.
* `goBackwards`: Whether the input sequence will be processed backwards. Default is false.
* `wRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `uRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), applied the recurrent weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, SimpleRNN}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(SimpleRNN(8, activation = "relu", inputShape = Shape(4, 5)))
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
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import SimpleRNN

model = Sequential()
model.add(SimpleRNN(8, activation = "relu", input_shape = (4, 5)))
input = np.random.random([3, 4, 5])
output = model.forward(input)
```
Input is:
```python
[[[0.84667859 0.46843956 0.58500761 0.49466075 0.80869937]
  [0.33858098 0.49313598 0.11871868 0.40802442 0.33438503]
  [0.76465152 0.70995096 0.35533213 0.76722302 0.92509398]
  [0.50201632 0.50159765 0.79646517 0.0496403  0.1004305 ]]

 [[0.73492087 0.60507454 0.50498809 0.57438087 0.45531948]
  [0.99241301 0.93581154 0.26433406 0.90715382 0.98946954]
  [0.5146376  0.34173451 0.52074524 0.0618145  0.35984059]
  [0.65040384 0.04413242 0.37820717 0.24391006 0.95041634]]

 [[0.87342021 0.8067062  0.62405708 0.85380849 0.0204989 ]
  [0.43654301 0.25973309 0.03199391 0.79672145 0.65800269]
  [0.31261367 0.686876   0.0629408  0.23044748 0.3881871 ]
  [0.2636806  0.37639863 0.64793255 0.84616263 0.66620196]]]
```
Output is:
```python
[[0.1865956  0.286946   0.03105135 0.5189484  0.63897055 0.0  0.40103152 0.5835867 ]
 [0.15814234 0.15947476 0.0        0.5763033  0.59285486 0.0  0.33782864 0.0       ]
 [0.43670028 0.32656467 0.27376643 0.5266341  0.4105177  0.0  0.07578468 0.34243858]]
```

---
## **GRU**
Gated Recurrent Unit architecture.

The input of this layer should be 3D, i.e. (batch, time steps, input dim).

**Scala:**
```scala
GRU(outputDim, activation = null, innerActivation = null, returnSequences = false, goBackwards = false, wRegularizer = null, uRegularizer = null, bRegularizer = null, inputShape = null)
```
**Python:**
```python
GRU(output_dim, activation="tanh", inner_activation="hard_sigmoid", return_sequences=False, go_backwards=False, W_regularizer=None, U_regularizer=None, b_regularizer=None, input_shape=None)
```

Parameters:

* `outputDim`: Hidden unit size. Dimension of internal projections and final output.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is 'tanh'.
* `innerActivation`: String representation of the activation function for inner cells. See [here](activation/#available-activations) for available activation strings. Default is 'hard_sigmoid'.
* `returnSequences`: Whether to return the full sequence or only return the last output in the output sequence. Default is false.
* `goBackwards`: Whether the input sequence will be processed backwards. Default is false.
* `wRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `uRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), applied the recurrent weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GRU}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(GRU(8, inputShape = Shape(2, 3)))
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
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import GRU

model = Sequential()
model.add(GRU(8, input_shape = (2, 3)))
input = np.random.random([1, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.1933675  0.06995704 0.64020873]
  [0.53693786 0.52344421 0.70939187]]]
```
Output is:
```python
[[-0.28945404 -0.04611638 -0.3548464  -0.2695646  -0.01624351 -0.37065172 0.30741248 -0.12380156]]
```

---
## **LSTM**
Long Short Term Memory unit architecture.

The input of this layer should be 3D, i.e. (batch, time steps, input dim).

**Scala:**
```scala
LSTM(outputDim, activation = null, innerActivation = null, returnSequences = false, goBackwards = false, wRegularizer = null, uRegularizer = null, bRegularizer = null, inputShape = null)
```
**Python:**
```python
LSTM(output_dim, activation="tanh", inner_activation="hard_sigmoid", return_sequences=False, go_backwards=False, W_regularizer=None, U_regularizer=None, b_regularizer=None, input_shape=None, input_shape=None)
```

Parameters:

* `outputDim`: Hidden unit size. Dimension of internal projections and final output.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is 'tanh'.
* `innerActivation`: String representation of the activation function for inner cells. See [here](activation/#available-activations) for available activation strings. Default is 'hard_sigmoid'.
* `returnSequences`: Whether to return the full sequence or only return the last output in the output sequence. Default is false.
* `goBackwards`: Whether the input sequence will be processed backwards. Default is false.
* `wRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `uRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), applied the recurrent weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, LSTM}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(LSTM(8, inputShape = Shape(2, 3)))
val input = Tensor[Float](1, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-2.1447723	0.31525767	-1.5426548
-0.63483864	-0.92148876	-2.0270665

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
-0.059685573 -0.06620621 -0.11752512 -0.00423051 -0.36373916 0.17949651 -0.011495307 -0.3951684
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x8]

```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import LSTM

model = Sequential()
model.add(LSTM(8, input_shape = (2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.84004043 0.2081865  0.76093342]
  [0.06878797 0.13804673 0.23251666]]

 [[0.24651173 0.5650254  0.41424478]
  [0.49338729 0.40505622 0.01497762]]]
```
Output is:
```python
[[ 0.01089199  0.02563154 -0.04335827  0.03037791  0.11265078 -0.17756112
   0.14166507  0.01017009]
 [ 0.0144811   0.03360332  0.00676281 -0.01473055  0.09639315 -0.16620669
   0.07391933  0.01746811]]
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
Highway(activation=None, W_regularizer=None, b_regularizer=None, bias=True, input_shape=None)
```

Parameters:

* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `wRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Highway}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(Highway(activation = "tanh", bias = false, inputShape = Shape(4)))
val input = Tensor[Float](2, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-1.1009767	-1.7566829	0.98709023	0.7114766
1.1036539	-0.6377753	-1.9605356	-2.4905455
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
-0.9396665	0.12595776	0.41412824	0.5987828
0.90716076	-0.54083437	-1.5220025	-1.3661726
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Highway

model = Sequential()
model.add(Highway(input_shape = (3)))
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

---
