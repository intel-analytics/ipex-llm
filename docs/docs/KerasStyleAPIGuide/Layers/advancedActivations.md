---
## **ELU**
Exponential Linear Unit.

It follows: f(x) =  alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0.

**Scala:**
```scala
ELU(alpha = 1.0, inputShape = null)
```
**Python:**
```python
ELU(alpha=1.0, input_shape=None)
```

**Parameters:**

* `alpha`: Double, scale for the negative factor.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, ELU}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(ELU(inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.13405465	0.05160992	-1.4711418
1.5808829	-1.3145303	0.6709266
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
-0.1254577	0.05160992	-0.77033687
1.5808829	-0.73139954	0.6709266
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import ELU

model = Sequential()
model.add(ELU(1.2, input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.98624221 0.93625254 0.51908689]
 [0.93353209 0.72231469 0.41468629]]
```
Output is
```python
[[0.98624223 0.93625253 0.5190869 ]
 [0.93353206 0.7223147  0.4146863 ]]
```

---
## **LeakyReLU**
Leaky version of a Rectified Linear Unit.

It allows a small gradient when the unit is not active: f(x) = alpha * x for x < 0, f(x) = x for x >= 0.

**Scala:**
```scala
LeakyReLU(alpha = 0.01, inputShape = null)
```
**Python:**
```python
LeakyReLU(alpha=0.01, input_shape=None)
```

**Parameters:**

* `alpha`: Double >= 0. Negative slope coefficient.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, LeakyReLU}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(LeakyReLU(1.27, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.5824592	-0.67302877	-1.7967145	0.071719564
0.6605261	0.08600852	0.30068552	0.11830905
-0.29188097	1.6119281	-0.9510023	-0.40019512

(2,.,.) =
0.39399254	-0.23085448	-1.2176765	-0.51953214
0.7793783	0.4649485	0.9937087	-0.8057158
-1.0953056	0.43886897	0.39663073	0.77146864

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.7397232	-0.8547465	-2.2818274	0.071719564
0.6605261	0.08600852	0.30068552	0.11830905
-0.37068883	1.6119281	-1.2077729	-0.5082478

(2,.,.) =
0.39399254	-0.29318517	-1.5464492	-0.65980583
0.7793783	0.4649485	0.9937087	-1.023259
-1.3910381	0.43886897	0.39663073	0.77146864

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import LeakyReLU

model = Sequential()
model.add(LeakyReLU(input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.93740357 0.60127816 0.90725787]
 [0.73843463 0.50795536 0.49504337]]
```
Output is
```python
[[0.93740356 0.6012782  0.90725785]
 [0.7384346  0.5079554  0.49504337]]
```

---
## **SReLU**
S-shaped Rectified Linear Unit.

It follows: f(x) = t^r + a^r(x - t^r) for x >= t^r, f(x) = x for t^r > x > t^l, f(x) = t^l + a^l(x - t^l) for x <= t^l.

**Scala:**
```scala
SReLU(tLeftInit = Zeros, aLeftInit = Xavier, tRightInit = Xavier, aRightInit = Ones, sharedAxes = null, inputShape = null)
```
**Python:**
```python
SReLU(t_left_init='zero', a_left_init='glorot_uniform', t_right_init='glorot_uniform', a_right_init='one', shared_axes=None, input_shape=None)
```

**Parameters:**

* `tLeftInit`: Initialization function for the left part intercept. Default is Zeros. You can also pass in corresponding string representations such as 'zero' or 'normal', etc. for simple init methods in the factory method.
* `aLeftInit`: Initialization function for the left part slope. Default is Xavier. You can also pass in corresponding string representations such as 'glorot_uniform', etc. for simple init methods in the factory method.
* `tRightInit`: Initialization function for the right part intercept. Default is Xavier. You can also pass in corresponding string representations such as 'glorot_uniform', etc. for simple init methods in the factory method.
* `aRightInit`: Initialization function for the right part slope. Default is Ones. You can also pass in corresponding string representations such as 'one' or 'normal', etc. for simple init methods in the factory method.
* `sharedAxes`: Array of Int. The axes along which to share learnable parameters for the activation function. Default is null.
For example, if the incoming feature maps are from a 2D convolution with output shape (batch, height, width, channels), and you wish to share parameters across space so that each filter only has one set of parameters, set 'SharedAxes = Array(1,2)'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, SReLU}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(SReLU(inputShape = Shape(2, 3)))
val input = Tensor[Float](2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.5599429	0.22811626	-0.027771426
-0.56582874	1.9261217	1.2686813

(2,.,.) =
0.7538568	0.8725621	0.19803657
0.49057	    0.0537252	0.8684544

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.5599429	0.22811626	-0.009864618
0.07011698	1.9261217	1.2686813

(2,.,.) =
0.7538568	0.87256205	0.19803657
0.49057	    0.0537252	0.8684544

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import SReLU

model = Sequential()
model.add(SReLU(input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.18934493 0.41314691 0.52979433 0.54267574]
  [0.40451538 0.17045234 0.01904761 0.11590762]
  [0.17007451 0.09039303 0.9700648  0.8030194 ]]

 [[0.42074793 0.00447267 0.36424064 0.92617442]
  [0.97258511 0.46134389 0.82389101 0.45722536]
  [0.97305309 0.76862034 0.71869518 0.88380866]]]
```
Output is
```python
[[[0.18934493 0.4131469  0.52979434 0.54267573]
  [0.4045154  0.17045234 0.01904761 0.11590762]
  [0.17007451 0.09039303 0.9700648  0.8030194 ]]

 [[0.42074794 0.00447267 0.36424065 0.9261744 ]
  [0.9725851  0.46134388 0.823891   0.45722535]
  [0.9730531  0.7686203  0.71869516 0.8838087 ]]]
```

---
## **ThresholdedReLU**
Thresholded Rectified Linear Unit.

It follows: f(x) = x for x > theta, f(x) = 0 otherwise.

**Scala:**
```scala
ThresholdedReLU(theta = 1.0, inputShape = null)
```
**Python:**
```python
ThresholdedReLU(theta=1.0, input_shape=None)
```

**Parameters:**

* `theta`: Double >= 0. Threshold location of activation.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, ThresholdedReLU}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(ThresholdedReLU(inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-2.1033075	1.4546076	-0.96647346	    -0.4511009
-2.4208899	-0.52082974	-0.081632175	-0.86960655
-0.6928835	2.7436976	1.538436	    -0.25845975

(2,.,.) =
0.34507668	-1.7959352	-0.62471783	    -0.75249046
-0.5291317	1.1868533	0.05719685	    0.8247873
-0.8376612	1.5174114	0.29486847	    -0.45751894

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.0	    1.4546076	0.0	        0.0
0.0	    0.0	        0.0	        0.0
0.0	    2.7436976	1.538436	0.0

(2,.,.) =
0.0	    0.0	        0.0	        0.0
0.0	    1.1868533	0.0	        0.0
0.0	    1.5174114	0.0	        0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import ThresholdedReLU

model = Sequential()
model.add(ThresholdedReLU(input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.91854565 0.58317415 0.33089385]
 [0.82472184 0.70572913 0.32803604]]
```
Output is
```python
[[0.0   0.0     0.0]
 [0.0   0.0     0.0]]
```

---
