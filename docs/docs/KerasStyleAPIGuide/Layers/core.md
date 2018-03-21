## **InputLayer**
Can be used as an entry point into a model.

**Scala:**
```scala
InputLayer(inputShape = null, name = null)
```
**Python:**
```python
InputLayer(input_shape=None, name=None)
```
Parameters:

* `inputShape`: For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.
* `name`: String to set the name of the input node. If not specified, its name will by default to be a generated string.


---
## **Dense**
A densely-connected NN layer.

The most common input is 2D.

**Scala:**
```scala
Dense(outputDim, init = "glorot_uniform", activation = null, wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
Dense(output_dim, init="glorot_uniform", activation=None, W_regularizer=None, b_regularizer=None, bias=True, input_dim=None, input_shape=None, name=None)
```

**Parameters:**

* `outputDim`: The size of the output dimension.
* `init`: String representation of the initialization method for the weights of the layer. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is "glorot_uniform".
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `wRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Dense}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(Dense(5, activation = "relu", inputShape = Shape(4)))
val input = Tensor[Float](2, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
 1.8646977	-0.059090078  0.091468036   0.6387431
-0.4485392	   1.5150243  -0.60176533  -0.6811443
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
1.0648216  0.0         0.0  0.0  0.0
0.0        0.20690927  0.0  0.0  0.34191078
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x5]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Dense

model = Sequential()
model.add(Dense(5, activation="relu", input_shape=(4, )))
input = np.random.random([2, 4])
output = model.forward(input)
```
Input is:
```python
[[ 0.26202468  0.15868397  0.27812652  0.45931689]
 [ 0.32100054  0.51839282  0.26194293  0.97608528]]
```
Output is
```python
[[ 0.  0.  0.  0.02094215  0.38839486]
 [ 0.  0.  0.  0.24498197  0.38024583]]
```

---
## **Flatten**
Flattens the input without affecting the batch size.

**Scala:**
```scala
Flatten(inputShape = null)
```
**Python:**
```python
Flatten(input_shape=None)
```

**Parameters:**

* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Flatten}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(Flatten(inputShape = Shape(2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
1.2196734	-0.37271047	-0.31215316
-0.68951845	-0.20356052	-0.85899264

(1,2,.,.) =
-1.7452804	-0.1138052	-0.9124519
-0.94204897	0.28943604	-0.71905166

(2,1,.,.) =
0.7228912	-0.51781553	-0.5869045
-0.82529205	0.26846665	-0.6199292

(2,2,.,.) =
-0.4529333	-0.57688874	0.9097755
0.7112487	-0.6711465	1.3074298

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
1.2196734	-0.37271047	-0.31215316	-0.68951845	-0.20356052	-0.85899264	-1.7452804	-0.1138052	-0.9124519	-0.94204897	0.28943604	-0.71905166
0.7228912	-0.51781553	-0.5869045	-0.82529205	0.26846665	-0.6199292	-0.4529333	-0.57688874	0.9097755	0.7112487	-0.6711465	1.3074298
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x12]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Flatten

model = Sequential()
model.add(Flatten(input_shape=(2, 4)))
input = np.random.random([2, 2, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.23566997 0.89368838 0.21225493 0.08271606]
  [0.56553029 0.94980164 0.27770336 0.66696008]]

 [[0.45052917 0.23468265 0.81973834 0.32676311]
  [0.80632879 0.72353573 0.92546756 0.08359752]]]
```
Output is
```python
[[0.23566997 0.8936884  0.21225493 0.08271606 0.5655303  0.9498016 0.27770337 0.66696006]
 [0.45052916 0.23468265 0.8197383  0.32676312 0.8063288  0.7235357 0.92546755 0.08359752]]
```

---
## **Reshape**
Reshapes an output to a certain shape.

Supports shape inference by allowing one -1 in the target shape. For example, if inputShape is (2, 3, 4), targetShape is (3, -1), then outputShape will be (3, 8).

**Scala:**
```scala
Reshape(targetShape, inputShape = null)
```
**Python:**
```python
Reshape(target_shape, input_shape=None)
```

**Parameters:**

* `targetShape`: Array of int. The target shape that you desire to have. Batch dimension should be excluded.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Reshape}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(Reshape(Array(3, 8), inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-1.7092276	-1.3941092	-0.6348466	0.71309644
0.3605411	0.025597548	0.4287048	-0.548675
0.4623341	-2.3912702	0.22030865	-0.058272455

(1,2,.,.) =
-1.5049093	-1.8828062	0.8230564	-0.020209199
-0.3415721	1.1219939	1.1089007	-0.74697906
-1.503861	-1.616539	0.048006497	1.1613717

(2,1,.,.) =
0.21216023	1.0107462	0.8586909	-0.05644316
-0.31436008	1.6892323	-0.9961186	-0.08169463
0.3559391	0.010261055	-0.70408463	-1.2480727

(2,2,.,.) =
1.7663039	0.07122444	0.073556066	-0.7847014
0.17604464	-0.99110585	-1.0302067	-0.39024687
-0.0260166	-0.43142694	0.28443158	0.72679126

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-1.7092276	-1.3941092	-0.6348466	0.71309644	0.3605411	0.025597548	0.4287048	-0.548675
0.4623341	-2.3912702	0.22030865	-0.058272455	-1.5049093	-1.8828062	0.8230564	-0.020209199
-0.3415721	1.1219939	1.1089007	-0.74697906	-1.503861	-1.616539	0.048006497	1.1613717

(2,.,.) =
0.21216023	1.0107462	0.8586909	-0.05644316	-0.31436008	1.6892323	-0.9961186	-0.08169463
0.3559391	0.010261055	-0.70408463	-1.2480727	1.7663039	0.07122444	0.073556066	-0.7847014
0.17604464	-0.99110585	-1.0302067	-0.39024687	-0.0260166	-0.43142694	0.28443158	0.72679126

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x8]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Reshape

model = Sequential()
model.add(Reshape(target_shape=(4, 2), input_shape=(2, 4)))
input = np.random.random([2, 2, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.03325335 0.39134471 0.8467274  0.40617943]
  [0.97296663 0.53771664 0.68796765 0.64503305]]

 [[0.65894994 0.47462534 0.57494741 0.05268725]
  [0.31745356 0.74422134 0.07159646 0.68379332]]]
```
Output is
```python
[[[0.03325335 0.3913447 ]
  [0.8467274  0.40617943]
  [0.9729666  0.5377166 ]
  [0.68796766 0.64503306]]

 [[0.65895    0.47462535]
  [0.5749474  0.05268725]
  [0.31745356 0.7442213 ]
  [0.07159646 0.6837933 ]]]
  [0.07159646 0.6837933 ]]]
```

---
## **Permute**
Permutes the dimensions of the input according to a given pattern.

Useful for connecting RNNs and convnets together.

**Scala:**
```scala
Permute(dims, inputShape = null)
```
**Python:**
```python
Permute(dims, input_shape=None)
```

**Parameters:**

* `dims`: Int array. Permutation pattern, does not include the samples dimension. Indexing starts at 1.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Permute}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(Permute(Array(2, 3, 1), inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-1.1030567	-1.4624393	0.6139582	0.21287616
-2.2278674	-2.5211496	1.9219213	0.85134244
0.32953477	-2.1209111	-0.82459116	-0.82447577

(1,2,.,.) =
1.0540756	2.2638302	0.19139263	-0.9037997
-0.20562297	-0.07835103	0.3883783	0.20750551
-0.56583923	0.9617757	-0.5792387	0.9008493

(2,1,.,.) =
-0.54270995	-1.9089237	0.9289245	0.27833897
-1.4734148	-0.9408616	-0.40362656	-1.1730295
0.9813707	-0.0040280274	-1.5321463	-1.4322052

(2,2,.,.) =
-0.056844145	2.2309854	2.1172705	0.10043324
1.121064	0.16069101	-0.51750094	-1.9682871
0.9011646	0.47903928	-0.54172426	-0.6604068

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
-1.1030567	1.0540756
-1.4624393	2.2638302
0.6139582	0.19139263
0.21287616	-0.9037997

(1,2,.,.) =
-2.2278674	-0.20562297
-2.5211496	-0.07835103
1.9219213	0.3883783
0.85134244	0.20750551

(1,3,.,.) =
0.32953477	-0.56583923
-2.1209111	0.9617757
-0.82459116	-0.5792387
-0.82447577	0.9008493

(2,1,.,.) =
-0.54270995	-0.056844145
-1.9089237	2.2309854
0.9289245	2.1172705
0.27833897	0.10043324

(2,2,.,.) =
-1.4734148	1.121064
-0.9408616	0.16069101
-0.40362656	-0.51750094
-1.1730295	-1.9682871

(2,3,.,.) =
0.9813707	0.9011646
-0.0040280274	0.47903928
-1.5321463	-0.54172426
-1.4322052	-0.6604068

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4x2]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Permute

model = Sequential()
model.add(Permute(dims=(2, 1), input_shape=(2, 4)))
input = np.random.random([2, 2, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.63966353 0.79842335 0.2066892  0.56806715]
  [0.1639401  0.61301646 0.81731068 0.53734401]]

 [[0.87178529 0.51120426 0.94765845 0.38695451]
  [0.32721816 0.19257422 0.44084815 0.65827817]]]
```
Output is
```python
[[[0.6396635  0.1639401 ]
  [0.79842335 0.6130165 ]
  [0.2066892  0.8173107 ]
  [0.56806713 0.537344  ]]

 [[0.8717853  0.32721817]
  [0.51120424 0.19257422]
  [0.9476585  0.44084814]
  [0.38695452 0.65827817]]]
```

---
## **RepeatVector**
Repeats the input n times.

The input of this layer should be 2D.

**Scala:**
```scala
RepeatVector(n, inputShape = null)
```
**Python:**
```python
RepeatVector(n, input_shape=None)
```

**Parameters:**

* `n`: Repetition factor. Integer.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, RepeatVector}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(RepeatVector(4, inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.4182444	2.858577	1.3975657
-0.19606766	0.8585809	0.3027246
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
1.4182444	2.858577	1.3975657
1.4182444	2.858577	1.3975657
1.4182444	2.858577	1.3975657
1.4182444	2.858577	1.3975657

(2,.,.) =
-0.19606766	0.8585809	0.3027246
-0.19606766	0.8585809	0.3027246
-0.19606766	0.8585809	0.3027246
-0.19606766	0.8585809	0.3027246

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import RepeatVector

model = Sequential()
model.add(RepeatVector(2, input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.62618263 0.06431292 0.42046559]
 [0.83101863 0.53810303 0.47986746]]
```
Output is
```python
[[[0.6261826  0.06431292 0.4204656 ]
  [0.6261826  0.06431292 0.4204656 ]]

 [[0.8310186  0.53810304 0.47986746]
  [0.8310186  0.53810304 0.47986746]]]
```

---
## **Merge**
Used to merge a list of tensors into a single tensor, following some merge mode.

Merge must have at least two input layers.

**Scala:**
```scala
Merge(layers = null, mode = "sum", concatAxis = -1, inputShape = null)
```
**Python:**
```python
Merge(layers=None, mode="sum", concat_axis=-1, input_shape=None)
```

**Parameters:**

* `layers`: A list of layer instances. Must be more than one layer.
* `mode`: Merge mode. String, must be one of: 'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'. Default is 'sum'.
* `concatAxis`: Integer, axis to use when concatenating layers. Only specify this when merge mode is 'concat'. Default is -1, meaning the last axis of the input.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Merge, InputLayer}
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.bigdl.tensor._

val input1 = Tensor[Float](2, 2, 3).rand(0, 1)
val input2 = Tensor[Float](2, 2, 3).rand(0, 1)
val input = T(1 -> input1, 2 -> input2)
val model = Sequential[Float]()
val l1 = InputLayer[Float](inputShape = Shape(2, 3))
val l2 = InputLayer[Float](inputShape = Shape(2, 3))
val layer = Merge[Float](layers = List(l1, l2), mode = "sum")
model.add(layer)
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.utils.Table =
 {
	2: (1,.,.) =
	   0.87815475	0.15025006	0.34412447
	   0.07909282	0.008027249	0.111715704

	   (2,.,.) =
	   0.52245367	0.2547527	0.35857987
	   0.7718501	0.26783863	0.8642062

	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
	1: (1,.,.) =
	   0.5377018	0.28364193	0.3424284
	   0.0075349305	0.9018168	0.9435114

	   (2,.,.) =
	   0.09112563	0.88585275	0.3100201
	   0.7910178	0.57497376	0.39764535

	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
 }
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
1.4158566	0.433892	0.6865529
0.08662775	0.90984404	1.0552272

(2,.,.) =
0.6135793	1.1406054	0.66859996
1.5628679	0.8428124	1.2618515

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Merge, InputLayer

model = Sequential()
l1 = InputLayer(input_shape=(3, 4))
l2 = InputLayer(input_shape=(3, 4))
model.add(Merge(layers=[l1, l2], mode='sum'))
input = [np.random.random([2, 3, 4]), np.random.random([2, 3, 4])]
output = model.forward(input)
```
Input is:
```python
[array([[[0.28764351, 0.0236015 , 0.78927442, 0.52646492],
        [0.63922826, 0.45101604, 0.4555552 , 0.70105653],
        [0.75790798, 0.78551523, 0.00686686, 0.61290369]],

       [[0.00430865, 0.3303661 , 0.59915782, 0.90362298],
        [0.26230717, 0.99383052, 0.50630521, 0.99119486],
        [0.56138318, 0.68165639, 0.10644523, 0.51860127]]]),

 array([[[0.84365767, 0.8854741 , 0.84183673, 0.96322321],
        [0.49354248, 0.97936826, 0.2266097 , 0.88083622],
        [0.11011776, 0.65762034, 0.17446099, 0.76658969]],

       [[0.58266689, 0.86322199, 0.87122999, 0.19031255],
        [0.42275118, 0.76379413, 0.21355413, 0.81132937],
        [0.97294728, 0.68601731, 0.39871792, 0.63172344]]])]
```
Output is
```python
[[[1.1313012  0.90907556 1.6311111  1.4896882 ]
  [1.1327708  1.4303843  0.6821649  1.5818927 ]
  [0.8680257  1.4431355  0.18132785 1.3794935 ]]

 [[0.5869755  1.1935881  1.4703878  1.0939355 ]
  [0.68505836 1.7576246  0.71985936 1.8025242 ]
  [1.5343305  1.3676738  0.50516313 1.1503248 ]]]
```

---
