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
* `init`: String representation of the initialization method for the weights of the layer. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'glorot_uniform'.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Dense}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Dense(5, activation = "relu", inputShape = Shape(4)))
val input = Tensor[Float](2, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
 1.8646977	-0.059090078  0.091468036   0.6387431
-0.4485392	1.5150243     -0.60176533   -0.6811443
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
[[ 0.0   0.0     0.0     0.02094215  0.38839486]
 [ 0.0   0.0     0.0     0.24498197  0.38024583]]
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
Flatten(input_shape=None, name=None)
```

**Parameters:**

* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Flatten}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
model.add(Flatten(input_shape=(2, 2, 3)))
input = np.random.random([2, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[[0.86901694 0.18961039 0.40317114]
   [0.03546013 0.44338256 0.14267447]]
  [[0.08971508 0.04943281 0.47568212]
   [0.21874466 0.54040762 0.19513549]]]

 [[[0.89994454 0.10154699 0.19762439]
   [0.90341835 0.44006613 0.08758557]]
  [[0.51165122 0.15523108 0.47434121]
   [0.24526962 0.79663289 0.52078471]]]]
```
Output is
```python
[[0.86901695 0.18961039 0.40317115 0.03546013 0.44338256 0.14267448
  0.08971508 0.04943281 0.4756821  0.21874467 0.5404076  0.19513549]
 [0.89994454 0.10154699 0.1976244  0.90341836 0.44006613 0.08758558
  0.5116512  0.15523107 0.4743412  0.24526963 0.7966329  0.52078474]]
```

---
## **Reshape**
Reshapes an output to a certain shape.

Supports shape inference by allowing one -1 in the target shape. For example, if input shape is (2, 3, 4), target shape is (3, -1), then output shape will be (3, 8).

**Scala:**
```scala
Reshape(targetShape, inputShape = null)
```
**Python:**
```python
Reshape(target_shape, input_shape=None, name=None)
```

**Parameters:**

* `targetShape`: The target shape that you desire to have. Batch dimension should be excluded.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Reshape}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
-1.7092276	-1.3941092	-0.6348466	0.71309644	    0.3605411	0.025597548	0.4287048	-0.548675
0.4623341	-2.3912702	0.22030865	-0.058272455	-1.5049093	-1.8828062	0.8230564	-0.020209199
-0.3415721	1.1219939	1.1089007	-0.74697906	    -1.503861	-1.616539	0.048006497	1.1613717

(2,.,.) =
0.21216023	1.0107462	0.8586909	-0.05644316	    -0.31436008	1.6892323	-0.9961186	-0.08169463
0.3559391	0.010261055	-0.70408463	-1.2480727	    1.7663039	0.07122444	0.073556066	-0.7847014
0.17604464	-0.99110585	-1.0302067	-0.39024687	    -0.0260166	-0.43142694	0.28443158	0.72679126

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x8]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Reshape

model = Sequential()
model.add(Reshape(target_shape=(3, 8), input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.39260304 0.10383185 0.87490319 0.89167328]
   [0.61649117 0.43285247 0.86851582 0.97743004]
   [0.90018969 0.04303951 0.74263493 0.14208656]]
  [[0.66193405 0.93432157 0.76160537 0.70437459]
   [0.99953431 0.23016734 0.42293405 0.66078049]
   [0.03357645 0.9695145  0.30111138 0.67109948]]]

 [[[0.39640201 0.92930203 0.86027666 0.13958544]
   [0.34584767 0.14743425 0.93804016 0.38053062]
   [0.55068792 0.77375329 0.84161166 0.48131356]]
  [[0.90116368 0.53253689 0.03332962 0.58278686]
   [0.34935685 0.32599554 0.97641892 0.57696434]
   [0.53974677 0.90682861 0.20027319 0.05962118]]]]
```
Output is
```python
[[[0.39260304 0.10383185 0.8749032  0.89167327 0.6164912  0.43285248 0.86851585 0.97743005]
  [0.9001897  0.04303951 0.74263495 0.14208655 0.661934   0.9343216  0.7616054  0.7043746 ]
  [0.9995343  0.23016734 0.42293406 0.6607805  0.03357645 0.9695145  0.30111137 0.6710995 ]]

 [[0.396402   0.92930204 0.86027664 0.13958544 0.34584767 0.14743425 0.93804014 0.38053063]
  [0.5506879  0.7737533  0.8416117  0.48131356 0.9011637  0.53253686 0.03332962 0.58278686]
  [0.34935686 0.32599553 0.9764189  0.5769643  0.53974676 0.9068286  0.20027319 0.05962119]]]
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
Permute(dims, input_shape=None, name=None)
```

**Parameters:**

* `dims`: Permutation pattern, does not include the batch dimension. Indexing starts at 1.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Permute}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
-0.54270995	-1.9089237	   0.9289245	0.27833897
-1.4734148	-0.9408616	   -0.40362656	-1.1730295
0.9813707	-0.0040280274  -1.5321463	-1.4322052

(2,2,.,.) =
-0.056844145   2.2309854	2.1172705	  0.10043324
1.121064	   0.16069101	-0.51750094	  -1.9682871
0.9011646	   0.47903928	-0.54172426	  -0.6604068

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
0.9813707	   0.9011646
-0.0040280274  0.47903928
-1.5321463	   -0.54172426
-1.4322052	   -0.6604068

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4x2]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Permute

model = Sequential()
model.add(Permute(dims=(2, 3, 1), input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.47372355 0.18103412 0.07076151 0.51208742]
   [0.3830121  0.2036672  0.24978515 0.3458438 ]
   [0.34180976 0.54635229 0.90048856 0.89178666]]
  [[0.15893009 0.62223068 0.1060953  0.26898095]
   [0.97659789 0.72022333 0.12613522 0.66538681]
   [0.79589927 0.32906473 0.27806256 0.99698214]]]

 [[[0.14608597 0.96667223 0.17876087 0.37672275]
   [0.89726934 0.09588159 0.19987136 0.99728596]
   [0.592439   0.40126537 0.18349086 0.88102044]]
  [[0.29313258 0.94066727 0.57244849 0.79352687]
   [0.31302252 0.65390325 0.54829736 0.63749209]
   [0.76679177 0.43937809 0.06966902 0.27204878]]]]
```
Output is
```python
[[[[0.47372353 0.1589301 ]
   [0.18103412 0.6222307 ]
   [0.07076152 0.1060953 ]
   [0.5120874  0.26898095]]
  [[0.38301212 0.9765979 ]
   [0.2036672  0.7202233 ]
   [0.24978516 0.12613523]
   [0.3458438  0.6653868 ]]
  [[0.34180975 0.7958993 ]
   [0.54635227 0.32906473]
   [0.90048856 0.27806255]
   [0.89178663 0.99698216]]]

 [[[0.14608598 0.29313257]
   [0.96667224 0.9406673 ]
   [0.17876087 0.5724485 ]
   [0.37672275 0.7935269 ]]

  [[0.8972693  0.31302252]
   [0.09588159 0.65390325]
   [0.19987136 0.54829735]
   [0.99728596 0.63749206]]

  [[0.592439   0.76679176]
   [0.40126538 0.43937808]
   [0.18349086 0.06966902]
   [0.8810204  0.27204877]]]]
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
RepeatVector(n, input_shape=None, name=None)
```

**Parameters:**

* `n`: Repetition factor. Integer.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, RepeatVector}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
model.add(RepeatVector(4, input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.51416513 0.87768557 0.48015041]
 [0.66598164 0.58916225 0.03983186]]
```
Output is
```python
[[[0.5141651  0.87768555 0.4801504 ]
  [0.5141651  0.87768555 0.4801504 ]
  [0.5141651  0.87768555 0.4801504 ]
  [0.5141651  0.87768555 0.4801504 ]]

 [[0.66598165 0.58916223 0.03983186]
  [0.66598165 0.58916223 0.03983186]
  [0.66598165 0.58916223 0.03983186]
  [0.66598165 0.58916223 0.03983186]]]
```

---
## **Merge**
Used to merge a list of inputs into a single output, following some merge mode.

Merge must have at least two input layers.

**Scala:**
```scala
Merge(layers = null, mode = "sum", concatAxis = -1, inputShape = null)
```
**Python:**
```python
Merge(layers=None, mode="sum", concat_axis=-1, input_shape=None, name=None)
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
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
val l1 = InputLayer[Float](inputShape = Shape(2, 3))
val l2 = InputLayer[Float](inputShape = Shape(2, 3))
val layer = Merge[Float](layers = List(l1, l2), mode = "sum")
model.add(layer)
val input1 = Tensor[Float](2, 2, 3).rand(0, 1)
val input2 = Tensor[Float](2, 2, 3).rand(0, 1)
val input = T(1 -> input1, 2 -> input2)
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
## **Masking**
Use a mask value to skip timesteps for a sequence.

**Scala:**
```scala
Masking(maskValue = 0.0, inputShape = null)
```
**Python:**
```python
Masking(mask_value=0.0, input_shape=None, name=None)
```

**Parameters:**

* `maskValue`: Mask value. For each timestep in the input (the second dimension), if all the values in the input at that timestep are equal to 'maskValue', then the timestep will masked (skipped) in all downstream layers.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Masking}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Masking(inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.00938185	-1.1461893	-1.0204586
0.24702129	-2.2756217	0.010394359
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.00938185	-1.1461893	-1.0204586
0.24702129	-2.2756217	0.010394359
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Masking

model = Sequential()
model.add(Masking(input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.59540156 0.24933489 0.04434161]
 [0.89243422 0.68499562 0.36788333]]
```
Output is
```python
[[0.5954016  0.24933489 0.04434161]
 [0.89243424 0.68499565 0.36788332]]
```

---
## **MaxoutDense**
A dense maxout layer that takes the element-wise maximum of linear layers.

This allows the layer to learn a convex, piecewise linear activation function over the inputs.

The input of this layer should be 2D.

**Scala:**
```scala
MaxoutDense(outputDim, nbFeature = 4, wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
MaxoutDense(output_dim, nb_feature=4, W_regularizer=None, b_regularizer=None, bias=True, input_dim=None, input_shape=None, name=None)
```

**Parameters:**

* `outputDim`: The size of output dimension.
* `nbFeature`: Number of Dense layers to use internally. Integer. Default is 4.
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, MaxoutDense}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(MaxoutDense(2, inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-1.3550005	-1.1668127	-1.2882779
0.83600295	-1.94683	1.323666
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.71675766	1.2987505
0.9871184	0.6634239
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import MaxoutDense

model = Sequential()
model.add(MaxoutDense(2, input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.15996114 0.8391686  0.81922903]
 [0.52929427 0.35061754 0.88167693]]
```
Output is
```python
[[0.4479192  0.4842512]
 [0.16833156 0.521764 ]]
```