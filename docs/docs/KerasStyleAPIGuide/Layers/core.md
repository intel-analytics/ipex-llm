## **SparseDense**
SparseDense is the sparse version of layer Dense. SparseDense has two different from Dense:
firstly, SparseDense's input Tensor is a SparseTensor. Secondly, SparseDense doesn't backward
gradient to next layer in the backpropagation by default, as the gradInput of SparseDense is
useless and very big in most cases.

But, considering model like Wide&Deep, we provide backwardStart and backwardLength to backward
part of the gradient to next layer.

The most common input is 2D.

**Scala:**
```scala
SparseDense(outputDim, init = "glorot_uniform", activation = null, wRegularizer = null, bRegularizer = null, backwardStart = -1, backwardLength = -1, initWeight = null, initBias = null, initGradWeight = null, initGradBias = null, bias = true, inputShape = null)
```
**Python:**
```python
SparseDense(output_dim, init="glorot_uniform", activation=None, W_regularizer=None, b_regularizer=None, backward_start=-1, backward_length=-1, init_weight=None, init_bias=None, init_grad_weight=None, init_grad_bias=None, bias=True, input_shape=None, name=None)
```

**Parameters:**

* `outputDim`: The size of the output dimension.
* `init`: String representation of the initialization method for the weights of the layer. Default is 'glorot_uniform'.
* `activation`: String representation of the activation function to use. Default is null.
* `wRegularizer`: An instance of [Regularizer], applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer], applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `backwardStart`: Backward start index, counting from 1.
* `backwardLength`: Backward length.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a `Shape` object. For Python API, it should be a shape tuple. Batch dimension should be excluded.
* `name`: String to set the name of the layer. If not specified, its name will by default to be a generated string.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.SparseDense
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val layer = SparseDense[Float](outputDim = 5, inputShape = Shape(2, 4))
layer.build(Shape(-1, 2, 4))
val input = Tensor[Float](Array(2, 4)).rand()
input.setValue(1, 1, 1f)
input.setValue(2, 3, 3f)
val sparseInput = Tensor.sparse(input)
val output = layer.forward(sparseInput)
```
Input is:
```scala
input: 
(0, 0) : 1.0
(0, 1) : 0.2992794
(0, 2) : 0.11227019
(0, 3) : 0.722947
(1, 0) : 0.6147614
(1, 1) : 0.4288646
(1, 2) : 3.0
(1, 3) : 0.7749917
[com.intel.analytics.bigdl.tensor.SparseTensor of size 2x4]
```
Output is:
```scala
output: 
0.053516	0.33429605	0.22587383	-0.8998945	0.24308181	
0.76745665	-1.614114	0.5381658	-2.2226436	-0.15573677	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x5]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import SparseDense
from zoo.pipeline.api.keras.models import Sequential
from bigdl.util.common import JTensor

model = Sequential()
model.add(SparseDense(output_dim=2, input_shape=(3, 4)))
input = JTensor.sparse(
    a_ndarray=np.array([1, 3, 2, 4]),
    i_ndarray = np.array([[0, 0, 1, 2],
             [0, 3, 2, 1]]),
    shape = np.array([3, 4])
)
output = model.forward(input)
```
Input is:
```python
JTensor: storage: [1. 3. 2. 4.], shape: [3 4] ,indices [[0 0 1 2]
 [0 3 2 1]], float
```
Output is
```python
[[ 1.57136     2.29596   ]
 [ 0.5791738  -1.6598101 ]
 [ 2.331141   -0.84687066]]
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
import com.intel.analytics.zoo.pipeline.api.keras.layers.Reshape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
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
from zoo.pipeline.api.keras.layers import Reshape
from zoo.pipeline.api.keras.models import Sequential

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
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`MultiShape`](../keras-api-scala/#shape) object. For Python API, it should be a list of shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.InputLayer
import com.intel.analytics.zoo.pipeline.api.keras.layers.Merge
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
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
from zoo.pipeline.api.keras.layers import Merge, InputLayer
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
l1 = InputLayer(input_shape=(3, 4))
l2 = InputLayer(input_shape=(3, 4))
model.add(Merge(layers=[l1, l2], mode='sum'))
input = [np.random.random([2, 3, 4]), np.random.random([2, 3, 4])]
output = model.forward(input)
```
Input is:
```python
[[[[0.28764351, 0.0236015 , 0.78927442, 0.52646492],
   [0.63922826, 0.45101604, 0.4555552 , 0.70105653],
   [0.75790798, 0.78551523, 0.00686686, 0.61290369]],

  [[0.00430865, 0.3303661 , 0.59915782, 0.90362298],
   [0.26230717, 0.99383052, 0.50630521, 0.99119486],
   [0.56138318, 0.68165639, 0.10644523, 0.51860127]]],

 [[[0.84365767, 0.8854741 , 0.84183673, 0.96322321],
   [0.49354248, 0.97936826, 0.2266097 , 0.88083622],
   [0.11011776, 0.65762034, 0.17446099, 0.76658969]],

  [[0.58266689, 0.86322199, 0.87122999, 0.19031255],
   [0.42275118, 0.76379413, 0.21355413, 0.81132937],
   [0.97294728, 0.68601731, 0.39871792, 0.63172344]]]]
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
* `wRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.MaxoutDense
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
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
from zoo.pipeline.api.keras.layers import MaxoutDense
from zoo.pipeline.api.keras.models import Sequential

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

---
## **Squeeze**
Delete the singleton dimension(s). The batch dimension needs to be unchanged.

For example, if input has size (2, 1, 3, 4, 1):

Squeeze(1) will give output size (2, 3, 4, 1),

Squeeze() will give output size (2, 3, 4)

**Scala:**
```scala
Squeeze(dims = null, inputShape = null)
```
**Python:**
```python
Squeeze(dim=None, input_shape=None, name=None)
```

**Parameters:**

* `dims`: The dimension(s) to squeeze. 0-based index. Cannot squeeze the batch dimension. The selected dimensions must be singleton, i.e. having size 1. Default is null, and in this case all the non-batch singleton dimensions will be deleted.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.Squeeze
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Squeeze[Float](1, inputShape = Shape(1, 1, 32)))
val input = Tensor[Float](1, 1, 1, 32).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.5521966       -1.2199087      0.365958        1.3845297       0.115254946     -0.20352958     2.4912808       0.987046        -0.2115477      3.0530396      -1.0043625      1.4688021       -1.2412603      -0.25383064     0.49164283      -0.40329486     0.26323202      0.7979045       0.025444122   0.47221214       1.3995043       0.48498031      -0.86961967     -0.058370713    -0.85965866     -1.2727696      0.45570874      0.73393697      0.2567143      1.4261572       -0.37773672     -0.7339463

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x1x32]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.5521966       -1.2199087      0.365958        1.3845297       0.115254946     -0.20352958     2.4912808       0.987046        -0.2115477      3.0530396      -1.0043625      1.4688021       -1.2412603      -0.25383064     0.49164283      -0.40329486     0.26323202      0.7979045       0.025444122   0.47221214       1.3995043       0.48498031      -0.86961967     -0.058370713    -0.85965866     -1.2727696      0.45570874      0.73393697      0.2567143      1.4261572       -0.37773672     -0.7339463

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x32]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Squeeze
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Squeeze(1, input_shape=(1, 1, 32)))
input = np.random.random([1, 1, 1, 32])
output = model.forward(input)
```
Input is:
```python
[[[[0.20585343, 0.47011701, 0.14553177, 0.93915599, 0.57234281,
    0.91631229, 0.32244256, 0.94243351, 0.86595631, 0.73916763,
    0.35898731, 0.65208275, 0.07935983, 0.89313423, 0.68601269,
    0.48919672, 0.28406399, 0.20962799, 0.88071757, 0.45501821,
    0.60931183, 0.46709718, 0.14218838, 0.42517758, 0.9149958 ,
    0.0843243 , 0.27302307, 0.75281922, 0.3688931 , 0.86913729,
    0.89774196, 0.77838838]]]]
```
Output is
```python
[[[0.20585343, 0.470117  , 0.14553176, 0.939156  , 0.5723428 ,
   0.9163123 , 0.32244256, 0.94243354, 0.8659563 , 0.73916763,
   0.3589873 , 0.65208274, 0.07935983, 0.89313424, 0.6860127 ,
   0.48919672, 0.284064  , 0.20962799, 0.8807176 , 0.45501822,
   0.6093118 , 0.46709716, 0.14218839, 0.42517757, 0.9149958 ,
   0.0843243 , 0.27302307, 0.75281924, 0.36889312, 0.8691373 ,
   0.897742  , 0.7783884 ]]]
```

---
## **BinaryThreshold**
Threshold the input.

If an input element is smaller than the threshold value, it will be replaced by 0; otherwise, it will be replaced by 1.

**Scala:**
```scala
BinaryThreshold(value = 1e-6, inputShape = null)
```
**Python:**
```python
BinaryThreshold(value=1e-6, input_shape=None, name=None)
```

**Parameters:**

* `value`: The threshold value to compare with. Default is 1e-6.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.BinaryThreshold
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(BinaryThreshold[Float](inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-1.1907398      -0.18995096     -2.0344417      -1.3789974
-1.8801064      -0.74757665     -0.4339697      0.0058485097
0.7012256       -0.6363152      2.0156987       -0.5512639

(1,2,.,.) =
-0.5251603      0.082127444     0.29550993      1.6357868
-1.3828015      -0.11842779     0.3316966       -0.14360528
0.21216457      -0.117370956    -0.12934707     -0.35854268

(2,1,.,.) =
-0.9071151      -2.8566089      -0.4796377      -0.915065
-0.8439908      -0.25404388     -0.39926198     -0.15191565
-1.0496653      -0.403675       -1.3591816      0.5311797

(2,2,.,.) =
0.53509855      -0.08892822     1.2196561       -0.62759316
-0.47476718     -0.43337926     -0.10406987     1.4035174
-1.7120812      1.1328355       0.9219375       1.3813454

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.0     0.0     0.0     0.0
0.0     0.0     0.0     1.0
1.0     0.0     1.0     0.0

(1,2,.,.) =
0.0     1.0     1.0     1.0
0.0     0.0     1.0     0.0
1.0     0.0     0.0     0.0

(2,1,.,.) =
0.0     0.0     0.0     0.0
0.0     0.0     0.0     0.0
0.0     0.0     0.0     1.0

(2,2,.,.) =
1.0     0.0     1.0     0.0
0.0     0.0     0.0     1.0
0.0     1.0     1.0     1.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import BinaryThreshold
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(BinaryThreshold(input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.39251031, 0.89560064, 0.94826505, 0.12508531],
   [0.32738299, 0.4342822 , 0.4460907 , 0.22411114],
   [0.57244209, 0.4662434 , 0.77926641, 0.37520199]],

  [[0.96505949, 0.83902339, 0.27306266, 0.02059265],
   [0.04738931, 0.46020822, 0.90156452, 0.28831492],
   [0.6897974 , 0.41503956, 0.01914761, 0.01319553]]],


 [[[0.27872031, 0.09379916, 0.19864558, 0.88633738],
   [0.52456666, 0.33489463, 0.68707086, 0.23693124],
   [0.47535936, 0.29494383, 0.83557663, 0.32998656]],

  [[0.3098337 , 0.36557965, 0.60374616, 0.09181418],
   [0.08452758, 0.07904724, 0.36483289, 0.76342299],
   [0.67345088, 0.34587093, 0.00292717, 0.65672907]]]]
```
Output is
```python
[[[[1., 1., 1., 1.],
   [1., 1., 1., 1.],
   [1., 1., 1., 1.]],

  [[1., 1., 1., 1.],
   [1., 1., 1., 1.],
   [1., 1., 1., 1.]]],


 [[[1., 1., 1., 1.],
   [1., 1., 1., 1.],
   [1., 1., 1., 1.]],

  [[1., 1., 1., 1.],
   [1., 1., 1., 1.],
   [1., 1., 1., 1.]]]]
```

---
## **Sqrt**
Applies an element-wise square root operation to the input.

**Scala:**
```scala
Sqrt(inputShape = null)
```
**Python:**
```python
Sqrt(input_shape=None, name=None)
```

**Parameters:**

* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.Sqrt
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Sqrt[Float](inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.6950394       0.5234307       1.7375475
0.25833175      0.02685826      -0.6046901
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.8336902       0.7234851       1.3181607
0.50826347      0.16388491      NaN
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Sqrt
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Sqrt(input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.2484558 , 0.65280218, 0.35286984],
 [0.19616094, 0.30966802, 0.82148169]]
```
Output is
```python
[[0.4984534 , 0.80796176, 0.5940285 ],
 [0.4429006 , 0.55647826, 0.9063563 ]]
```

---
## **MulConstant**
Multiply the input by a (non-learnable) scalar constant.

**Scala:**
```scala
MulConstant(constant, inputShape = null)
```
**Python:**
```python
MulConstant(constant, input_shape=None, name=None)
```

**Parameters:**

* `constant`: The scalar constant to be multiplied.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.MulConstant
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(MulConstant[Float](2.2, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.16873977     1.0812985       1.0942211       -0.67091423
1.0086882       0.5915831       0.26184535      -1.361431
1.5616825       -0.037591368    1.2794676       1.0692137

(2,.,.) =
0.29868057      -0.23266982     -0.7679556      -2.209848
-0.13954644     -0.1368473      -0.54510623     1.8397199
-0.58691734     -0.56410027     -1.5567777      0.050648995

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.3712275      2.3788567       2.4072864       -1.4760114
2.219114        1.3014828       0.57605976      -2.9951482
3.4357016       -0.08270101     2.8148286       2.3522704

(2,.,.) =
0.6570973       -0.5118736      -1.6895024      -4.8616657
-0.3070022      -0.30106407     -1.1992338      4.047384
-1.2912182      -1.2410206      -3.424911       0.11142779

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import MulConstant
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(MulConstant(2.2, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.39874191, 0.66634984, 0.23907766, 0.31587494],
  [0.78842014, 0.93057835, 0.80739529, 0.71541279],
  [0.2231424 , 0.3372844 , 0.94678072, 0.52928034]],

 [[0.60142458, 0.41221671, 0.00890549, 0.32069845],
  [0.51122554, 0.76280426, 0.87579418, 0.17182832],
  [0.54133184, 0.19814384, 0.92529327, 0.5616615 ]]]
```
Output is
```python
[[[0.8772322 , 1.4659697 , 0.5259709 , 0.6949249 ],
  [1.7345244 , 2.0472724 , 1.7762697 , 1.5739082 ],
  [0.4909133 , 0.7420257 , 2.0829177 , 1.1644168 ]],

 [[1.3231341 , 0.9068768 , 0.01959208, 0.7055366 ],
  [1.1246961 , 1.6781695 , 1.9267472 , 0.37802234],
  [1.19093   , 0.43591645, 2.0356452 , 1.2356553 ]]]
```

---
## **Scale**
Scale is the combination of CMul and CAdd.

Computes the element-wise product of the input and weight, with the shape of the weight "expand" to match the shape of the input.

Similarly, perform an expanded bias and perform an element-wise add.

**Scala:**
```scala
Scale(size, inputShape = null)
```
**Python:**
```python
Scale(size, input_shape=None, name=None)
```

**Parameters:**

* `size`: Size of the weight and bias.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.Scale
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
var array = Array(1, 2)
model.add(Scale[Float](array, inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.006399727    -0.06412822     -0.2334789
0.31029955      1.6557469       1.9614618
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.09936619      0.57585865      0.20324506
0.38537437      -0.8598822      -1.0186496
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Scale
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Scale((2, 1), input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.7242994 , 0.77888884, 0.71470432],
 [0.03058471, 0.00602764, 0.57513629]]
```
Output is
```python
[[1.0946966 , 1.1255064 , 1.0892813 ],
 [0.58151895, 0.5909191 , 0.37307182]]
```

---
## **Log**
Applies a log transformation to the input.

**Scala:**
```scala
Log(inputShape = null)
```
**Python:**
```python
Log(input_shape=None, name=None)
```

**Parameters:**

* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.Log
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Log[Float](inputShape = Shape(2, 4, 4)))
val input = Tensor[Float](1, 2, 4, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.38405678      -0.5502389      -0.383079       -0.988537
-0.6294056      -0.7838047      0.8747865       -1.0659786
-2.2445498      -0.5488076      -0.42898977     0.6916364
1.6542299       -0.9966279      -0.38244298     1.6954672

(1,2,.,.) =
0.43478605      -0.6678534      1.9530942       -0.5209587
0.12899925      0.20572199      2.0359943       0.55223215
0.65247816      0.8792108       -0.38860792     0.48663738
-1.0084358      0.31141177      0.69208467      0.48385203

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x4x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
-0.95696485     NaN     NaN     NaN
NaN     NaN     -0.13377543     NaN
NaN     NaN     NaN     -0.36869493
0.5033356       NaN     NaN     0.5279584

(1,2,.,.) =
-0.83290124     NaN     0.6694149       NaN
-2.0479486      -1.5812296      0.7109843       -0.5937868
-0.4269776      -0.12873057     NaN     -0.720236
NaN     -1.1666392      -0.36804697     -0.72597617

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x4x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Log
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Log(input_shape=(2, 4, 4)))
input = np.random.random([1, 2, 4, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.90127539, 0.9861594 , 0.04722941, 0.63719453],
   [0.46529477, 0.81511804, 0.24435558, 0.45003562],
   [0.15170845, 0.35157662, 0.0925214 , 0.63852947],
   [0.27817508, 0.42572846, 0.44363004, 0.03536394]],

  [[0.65027784, 0.00429838, 0.07434429, 0.18653305],
   [0.19659183, 0.66647529, 0.77821197, 0.65894478],
   [0.28212032, 0.52307663, 0.09589939, 0.71547588],
   [0.84344158, 0.25291738, 0.52145649, 0.82982377]]]]
```
Output is
```python
[[[[-0.10394441, -0.01393729, -3.0527387 , -0.45068032],
   [-0.76508415, -0.20442237, -1.4091308 , -0.79842854],
   [-1.8857948 , -1.0453277 , -2.3803153 , -0.44858742],
   [-1.2795045 , -0.85395354, -0.8127643 , -3.3420627 ]],

  [[-0.43035555, -5.4495163 , -2.5990484 , -1.6791469 ],
   [-1.6266255 , -0.4057522 , -0.25075635, -0.41711554],
   [-1.2654216 , -0.64802724, -2.3444557 , -0.33480743],
   [-0.1702646 , -1.3746924 , -0.6511295 , -0.1865419 ]]]]
```

---
## **Identity**
Identity just return the input to output.

It's useful in same parallel container to get an origin input.

**Scala:**
```scala
Identity(inputShape = null)
```
**Python:**
```python
Identity(input_shape=None, name=None)
```

**Parameters:**

* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.Identity
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Identity[Float](inputShape = Shape(4, 4)))
val input = Tensor[Float](3, 4, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.9601166       -0.86010313     0.0023731247    -0.81219757
1.1469674       -1.5375912      -1.5348053      -0.34829113
-1.236773       -0.7183283      -0.89256984     0.8605067
0.7937664       0.52992857      -1.6157389      0.36134166

(2,.,.) =
-0.44434744     -0.23848957     -0.01632014     -0.58109635
-0.19856784     -2.3421717      -0.5868049      -0.76775354
0.80254126      1.78778 -1.1835604      1.4489703
0.8731402       0.8906672       0.2800079       -0.6715317

(3,.,.) =
1.4093032       2.358169        -1.4620789      1.1904576
-0.18263042     -0.31869793     2.01061 1.2159953
-0.5801479      1.2949371       -0.7510707      -1.0707517
0.30815956      -1.161963       -0.26964024     -0.4759499

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
1.9601166       -0.86010313     0.0023731247    -0.81219757
1.1469674       -1.5375912      -1.5348053      -0.34829113
-1.236773       -0.7183283      -0.89256984     0.8605067
0.7937664       0.52992857      -1.6157389      0.36134166

(2,.,.) =
-0.44434744     -0.23848957     -0.01632014     -0.58109635
-0.19856784     -2.3421717      -0.5868049      -0.76775354
0.80254126      1.78778 -1.1835604      1.4489703
0.8731402       0.8906672       0.2800079       -0.6715317

(3,.,.) =
1.4093032       2.358169        -1.4620789      1.1904576
-0.18263042     -0.31869793     2.01061 1.2159953
-0.5801479      1.2949371       -0.7510707      -1.0707517
0.30815956      -1.161963       -0.26964024     -0.4759499

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Identity
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Identity(input_shape=(4, 4)))
input = np.random.random([3, 4, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.36751123, 0.92287101, 0.73894405, 0.33699379],
  [0.69405782, 0.9653215 , 0.2617223 , 0.68205229],
  [0.71455325, 0.99419333, 0.90886495, 0.10232991],
  [0.1644055 , 0.30013138, 0.98921154, 0.26803146]],

  [[0.35898357, 0.72067882, 0.13236563, 0.71935521],
   [0.30865626, 0.71098844, 0.86718946, 0.12531168],
   [0.84916882, 0.84221518, 0.52186664, 0.87239729],
   [0.50637899, 0.10890469, 0.86832705, 0.93581179]],

  [[0.19640105, 0.09341008, 0.12043328, 0.09261859],
   [0.66019486, 0.07251262, 0.80929761, 0.39094486],
   [0.63027391, 0.39537796, 0.55578905, 0.53933265],
   [0.13885559, 0.56695373, 0.17036027, 0.4577097 ]]]
```
Output is
```python
[[[0.36751124, 0.922871  , 0.73894405, 0.33699378],
  [0.6940578 , 0.9653215 , 0.2617223 , 0.6820523 ],
  [0.71455324, 0.9941933 , 0.908865  , 0.10232991],
  [0.1644055 , 0.30013138, 0.98921156, 0.26803148]],

 [[0.35898358, 0.7206788 , 0.13236563, 0.7193552 ],
  [0.30865628, 0.71098846, 0.86718947, 0.12531169],
  [0.84916884, 0.8422152 , 0.5218666 , 0.8723973 ],
  [0.506379  , 0.10890469, 0.868327  , 0.9358118 ]],

 [[0.19640104, 0.09341008, 0.12043328, 0.09261858],
  [0.6601949 , 0.07251262, 0.8092976 , 0.39094487],
  [0.63027394, 0.39537796, 0.55578905, 0.5393326 ],
  [0.13885559, 0.5669537 , 0.17036027, 0.4577097 ]]]
```

---
## **Select**
Select an index of the input in the given dim and return the subset part.

The batch dimension needs to be unchanged.

For example, if input is:

[[1, 2, 3], 
 [4, 5, 6]]

Select(1, 1) will give output [2 5]

Select(1, -1) will give output [3 6]

**Scala:**
```scala
Select(dim, index, inputShape = null)
```
**Python:**
```python
Select(dim, index, input_shape=None, name=None)
```

**Parameters:**

* `dim`: The dimension to select. 0-based index. Cannot select the batch dimension. -1 means the last dimension of the input.
* `index`: The index of the dimension to be selected. 0-based index. -1 means the last dimension of the input.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.Select
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Select[Float](1, 2, inputShape = Shape(3, 1, 3)))
val input = Tensor[Float](1, 3, 1, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-0.67646945     -0.5485965      -0.11103154
(1,2,.,.) =
-0.13488655     0.43843046      -0.04482145
(1,3,.,.) =
-0.18094881     0.19431554      -1.7624844
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x1x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.18094881     0.19431554      -1.7624844
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x3]
```

**Python example:**
```python
from zoo.pipeline.api.keras.layers import Select
from zoo.pipeline.api.keras.models import Sequential
import numpy as np

model = Sequential()
model.add(Select(1, 2, input_shape=(3, 1, 3)))
input = np.random.random([1, 3, 1, 3])
output = model.forward(input)
```
Input is:
```python
array([[[[0.53306099, 0.95147881, 0.15222129]],
        [[0.89604861, 0.90160974, 0.5230576 ]],
        [[0.70779386, 0.14438568, 0.37601195]]]])
```
Output is:
```python
array([[[0.7077939 , 0.14438568, 0.37601194]]], dtype=float32)
```

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
* `init`: Initialization method for the weights of the layer. Default is Xavier.You can also pass in corresponding string representations such as 'glorot_uniform' or 'normal', etc. for simple init methods in the factory method.
* `activation`: Activation function to use. Default is null.You can also pass in corresponding string representations such as 'relu'or 'sigmoid', etc. for simple activations in the factory method.
* `wRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.Dense
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Dense[Float](5, activation = "relu", inputShape = Shape(4)))
val input = Tensor[Float](2, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.4289935       -1.7659454      -0.08306135     -1.0153456
1.0191492       0.37392816      1.3076705       -0.19495767
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.5421522       0.49008092      0.0     0.0     0.0
0.07940009      0.0     0.12953377      0.0     0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x5]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Dense
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Dense(5, activation="relu", input_shape=(4, )))
input = np.random.random([2, 4])
output = model.forward(input)
```
Input is:
```python
array([[0.64593485, 0.67393322, 0.72505368, 0.04654095],
       [0.19430753, 0.47800889, 0.00743648, 0.6412403 ]])
```
Output is
```python
array([[0.        , 0.        , 1.2698183 , 0.        , 0.10656227],
       [0.        , 0.        , 0.6236721 , 0.00299606, 0.29664695]],
      dtype=float32)
```

---
## **Negative**
Computes the negative value of each element of the input.

**Scala:**
```scala
Negative(inputShape = null)
```
**Python:**
```python
Negative(input_shape=None, name=None)
```

**Parameters:**

* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.Negative
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Negative[Float](inputShape = Shape(2, 3)))
val input = Tensor[Float](2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.031705        -0.5723963      1.998631
-0.32908052     2.4069138       -2.4111257
(2,.,.) =
0.5355049       -1.4404331      -0.38116863
-0.45641592     -1.1485358      0.94766915
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-1.031705       0.5723963       -1.998631
0.32908052      -2.4069138      2.4111257
(2,.,.) =
-0.5355049      1.4404331       0.38116863
0.45641592      1.1485358       -0.94766915
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```

**Python example:**
```python
from zoo.pipeline.api.keras.layers import Negative
from zoo.pipeline.api.keras.models import Sequential
import numpy as np

model = Sequential()
model.add(Negative(input_shape=(2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
array([[[0.39261261, 0.03164615, 0.32179116],
        [0.11969367, 0.61610712, 0.42573733]],
       [[0.36794656, 0.90912174, 0.540356  ],
        [0.42667627, 0.04154093, 0.84692964]]])
```
Output is
```python
array([[[-0.3926126 , -0.03164615, -0.32179114],
        [-0.11969367, -0.6161071 , -0.42573732]],
       [[-0.36794657, -0.90912175, -0.540356  ],
        [-0.42667627, -0.04154094, -0.84692967]]], dtype=float32)
```

---
## **CAdd**
This layer has a bias with given size.

The bias will be added element-wise to the input.

If the element number of the bias matches the input, a simple element-wise addition will be done.

Or the bias will be expanded to the same size of the input.

The expand means repeat on unmatched singleton dimension (if some unmatched dimension isn't a singleton dimension, an error will be raised).


**Scala:**
```scala
CAdd(size, bRegularizer = null, inputShape = null)
```
**Python:**
```python
CAdd(size, b_regularizer=None, input_shape=None, name=None)
```

**Parameters:**

* `size`: the size of the bias
* `bRegularizer`: An instance of [Regularizer](https://bigdl-project.github.io/master/#APIGuide/Regularizers/), applied to the bias. Default is null.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.CAdd
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(CAdd[Float](Array(2, 3), inputShape = Shape(2, 3)))
val input = Tensor[Float](2, 2, 3).rand()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.2183351       0.32434112      0.89350265
0.3348259       0.78677046      0.24054797
(2,.,.) =
0.9945844       0.72363794      0.7737936
0.05522544      0.3517818       0.7417069
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.1358028       0.6956667       1.0837181
0.6767027       0.7955346       0.5063505
(2,.,.) =
0.9120521       1.0949634       0.96400905
0.3971022       0.36054593      1.0075095
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```

**Python example:**
```python
from zoo.pipeline.api.keras.layers import CAdd
from zoo.pipeline.api.keras.models import Sequential
import numpy as np

model = Sequential()
model.add(CAdd([2, 1], input_shape = (2, 3)))
input = np.random.rand(2, 2, 3)
output = model.forward(input)
```
Input is:
```python
array([[[0.4122004 , 0.73289359, 0.11500016],
        [0.26974491, 0.32166632, 0.91408442]],
       [[0.66824327, 0.80271314, 0.75981145],
        [0.39271431, 0.07312566, 0.4966805 ]]])
```
Output is
```python
array([[[ 0.06560206,  0.38629526, -0.23159817],
        [ 0.44287407,  0.4947955 ,  1.0872136 ]],
       [[ 0.32164496,  0.45611483,  0.41321313],
        [ 0.56584346,  0.24625483,  0.6698097 ]]], dtype=float32)
```

---
## **GaussianSampler**
Takes {mean, log_variance} as input and samples from the Gaussian distribution.

**Scala:**
```scala
GaussianSampler(inputShape = null)
```
**Python:**
```python
GaussianSampler(input_shape=None, name=None)
```

**Parameters:**

* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`MultiShape`](../keras-api-scala/#shape) object that consists of two identical Single Shape. For Python API, it should be a list of two identical shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.GaussianSampler
import com.intel.analytics.bigdl.utils.{Shape, MultiShape, T}
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
val shape1 = Shape(2, 3)
val shape2 = Shape(2, 3)
model.add(GaussianSampler[Float](inputShape = MultiShape(List(shape1,shape2))))
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
           0.9996127    0.8964211       0.7424038
           0.40628982   0.37035564      0.20108517

           (2,.,.) =
           0.6974727    0.60202897      0.1535999
           0.012422224  0.5993025       0.96206

           [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
        1: (1,.,.) =
           0.21060324   0.576583        0.21633287
           0.1484059    0.2730577       0.25317845

           (2,.,.) =
           0.58513683   0.58095694      0.18811373
           0.7029449    0.41235915      0.44636542

           [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
 }
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
1.5258198       1.9536011       -1.8591263
-1.0618867      -0.751225       0.35412917

(2,.,.) =
1.3334517       -0.60312974     0.7324476
0.09502721      0.8094909       0.44807082

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import GaussianSampler

model = Sequential()
model.add(GaussianSampler(input_shape=[(3,),(3,)]))
input1 = np.random.random([2, 3])
input2 = np.random.random([2, 3])
input = [input1, input2]
output = model.forward(input)
```
Input is:
```python
[[[0.79941342, 0.87462822, 0.9516901 ],
  [0.20111287, 0.54634077, 0.83614511]], 
  
 [[0.31886989, 0.22829382, 0.84355419],
  [0.51186641, 0.28043938, 0.29440057]]]
```
Output is
```python
[[ 0.71405387  2.2944303  -0.41778684]
 [ 0.84234     2.3337283  -0.18952972]]
```

---
## **Exp**
Applies element-wise exp to the input.

When you use this layer as the first layer of a model, you need to provide the argument inputShape (a Single Shape, does not include the batch dimension).

**Scala:**
```scala
Exp(inputShape = null)
```
**Python:**
```python
Exp(input_shape=None, name=None)
```

**Parameters:**

* `inputShape`: A Single Shape, does not include the batch dimension.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.Exp
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Exp[Float](inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-1.5841372      -0.13795324     -2.144475       0.09272669
1.055668        -1.2310301      1.2145554       -0.6073714
0.9296467       0.2923885       1.3364213       0.1652137

(1,2,.,.) =
0.2099718       -0.3856573      -0.92586        -0.5317779
0.6618383       -0.9677452      -1.5014665      -0.35464883
2.045924        -0.317644       -1.812726       0.95438373

(2,1,.,.) =
-0.4536791      -0.34785584     1.6424289       -0.07981159
-0.8022624      -0.4211059      0.3461831       1.9598864
-0.84695745     -0.6115283      0.7729755       2.3077402

(2,2,.,.) =
-0.08438411     -0.908458       0.6688936       -0.7292123
-0.26337254     0.55425745      -0.14925817     -0.010179609
-0.62562865     -1.0517743      -0.23839666     -1.144982

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.20512469      0.8711394       0.11712951      1.0971619
2.8738942       0.29199165      3.3687959       0.544781
2.533614        1.3396233       3.8054006       1.1796452

(1,2,.,.) =
1.2336433       0.6800035       0.39619055      0.5875594
1.9383523       0.37993878      0.22280318      0.7014197
7.7363033       0.7278619       0.16320862      2.5970695

(2,1,.,.) =
0.63528657      0.70620066      5.167706        0.92329025
0.44831353      0.6563206       1.4136615       7.0985208
0.42871734      0.5425211       2.1662023       10.051684

(2,2,.,.) =
0.9190782       0.4031454       1.9520763       0.48228875
0.76845556      1.740648        0.8613467       0.98987204
0.53492504      0.34931743      0.7878901       0.31822965

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import Exp

model = Sequential()
model.add(Exp(input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.93104587 0.94000338 0.84870765 0.98645553]
   [0.83708846 0.33375541 0.50119834 0.24879265]
   [0.51966475 0.84514791 0.15496452 0.61538968]]

  [[0.57250337 0.42520832 0.94850757 0.54317573]
   [0.64228691 0.9904079  0.01008592 0.51365217]
   [0.78640595 0.7717037  0.51277595 0.24245034]]]


 [[[0.82184752 0.92537331 0.20632728 0.47539445]
   [0.44604637 0.1507692  0.5437313  0.2074501 ]
   [0.93661363 0.93962609 0.29230559 0.74850958]]

  [[0.11659768 0.76177132 0.33194573 0.20695088]
   [0.49636212 0.85987328 0.49767861 0.96774006]
   [0.67669121 0.15542122 0.69981032 0.3349874 ]]]]
```
Output is
```python
[[[[2.5371614 2.5599902 2.3366253 2.6817122]
   [2.3096325 1.3962016 1.6506982 1.2824761]
   [1.6814638 2.3283222 1.1676165 1.8503776]]

  [[1.7726992 1.5299091 2.5818534 1.721465 ]
   [1.9008229 2.6923325 1.010137  1.6713842]
   [2.1954916 2.163449  1.6699204 1.2743679]]]


 [[[2.2746985 2.52281   1.2291554 1.6086487]
   [1.5621239 1.1627283 1.7224218 1.2305363]
   [2.551327  2.5590243 1.3395122 2.1138473]]

  [[1.1236672 2.1420672 1.3936772 1.2299222]
   [1.6427343 2.3628614 1.6448984 2.6319895]
   [1.9673574 1.16815   2.0133708 1.3979228]]]]
```

---
## **Square**
Applies an element-wise square operation to the input.

When you use this layer as the first layer of a model, you need to provide the argument inputShape (a Single Shape, does not include the batch dimension).

**Scala:**
```scala
Square(inputShape = null)
```
**Python:**
```python
Square(input_shape=None, name=None)
```

**Parameters:**

* `inputShape`: A Single Shape, does not include the batch dimension.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.Square
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Square[Float](inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-0.108013034    1.8879265       1.2232096       -1.5076439
1.4895755       -0.37966672     -0.34892964     0.15224025
-0.9296686      -1.1523775      0.14153497      -0.26954007

(1,2,.,.) =
-1.0875931      2.190617        -0.6903083      1.0039362
-0.1275677      -1.1096588      0.37359753      -0.17367937
0.23349741      0.14639114      -0.2330162      0.5343827

(2,1,.,.) =
0.3222191       0.21463287      -1.0157064      -0.22627507
1.1714277       0.43371263      1.069315        0.5122436
0.1958086       -1.4601041      2.5394423       -0.470833

(2,2,.,.) =
-0.38708544     -0.951611       -0.37234613     0.26813275
1.9477026       0.32779223      -1.2308712      -2.2376378
0.19652915      0.3304719       -1.7674786      -0.86961496

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.011666816     3.5642662       1.4962418       2.2729902
2.218835        0.14414681      0.1217519       0.023177093
0.86428374      1.3279738       0.020032147     0.07265185

(1,2,.,.) =
1.1828587       4.7988033       0.47652552      1.0078878
0.016273517     1.2313428       0.13957511      0.030164523
0.05452104      0.021430366     0.054296546     0.28556487

(2,1,.,.) =
0.10382515      0.046067268     1.0316595       0.05120041
1.3722429       0.18810664      1.1434345       0.26239353
0.038341008     2.131904        6.448767        0.22168371

(2,2,.,.) =
0.14983514      0.9055635       0.13864164      0.07189517
3.7935455       0.10744774      1.5150439       5.007023
0.038623706     0.109211676     3.1239805       0.7562302

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import Square

model = Sequential()
model.add(Square(input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.8708819  0.2698243  0.55854849 0.71699472]
   [0.66647234 0.72310216 0.8082119  0.66566951]
   [0.6714764  0.61394108 0.35063125 0.60473593]]

  [[0.37993365 0.64222557 0.96762005 0.18931697]
   [0.00529722 0.99133455 0.09786619 0.28988077]
   [0.60052911 0.83712995 0.59847519 0.54361243]]]


 [[[0.32832672 0.83316023 0.41272485 0.01963383]
   [0.89593955 0.73433713 0.67529323 0.69711912]
   [0.81251711 0.56755577 0.31958151 0.09795917]]

  [[0.46465895 0.22818875 0.31505317 0.41912166]
   [0.87865447 0.3799063  0.091204   0.68144165]
   [0.88274284 0.70479132 0.32074672 0.71771481]]]]
```
Output is
```python
[[[[7.5843531e-01 7.2805151e-02 3.1197643e-01 5.1408142e-01]
   [4.4418535e-01 5.2287674e-01 6.5320653e-01 4.4311589e-01]
   [4.5088059e-01 3.7692365e-01 1.2294226e-01 3.6570552e-01]]

  [[1.4434958e-01 4.1245368e-01 9.3628860e-01 3.5840917e-02]
   [2.8060573e-05 9.8274422e-01 9.5777912e-03 8.4030852e-02]
   [3.6063525e-01 7.0078653e-01 3.5817260e-01 2.9551446e-01]]]


 [[[1.0779844e-01 6.9415593e-01 1.7034180e-01 3.8548734e-04]
   [8.0270761e-01 5.3925103e-01 4.5602092e-01 4.8597506e-01]
   [6.6018403e-01 3.2211956e-01 1.0213234e-01 9.5959986e-03]]

  [[2.1590793e-01 5.2070107e-02 9.9258497e-02 1.7566296e-01]
   [7.7203369e-01 1.4432879e-01 8.3181690e-03 4.6436274e-01]
   [7.7923489e-01 4.9673077e-01 1.0287846e-01 5.1511449e-01]]]]
```
