## **ELU**
Exponential Linear Unit.

It follows: f(x) =  alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0.

**Scala:**
```scala
ELU(alpha = 1.0, inputShape = null)
```
**Python:**
```python
ELU(alpha=1.0, input_shape=None, name=None)
```

**Parameters:**

* `alpha`: Scale for the negative factor. Default is 1.0.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, ELU}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
model.add(ELU(input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.90404922 0.23530925 0.49711093]
 [0.43009161 0.22446032 0.90144771]]
```
Output is
```python
[[0.9040492  0.23530924 0.49711093]
 [0.43009162 0.22446032 0.9014477 ]]
```

---
## **LeakyReLU**
Leaky version of a Rectified Linear Unit.

It allows a small gradient when the unit is not active: f(x) = alpha * x for x < 0, f(x) = x for x >= 0.

**Scala:**
```scala
LeakyReLU(alpha = 0.3, inputShape = null)
```
**Python:**
```python
LeakyReLU(alpha=0.3, input_shape=None, name=None)
```

**Parameters:**

* `alpha`: Negative slope coefficient. Default is 0.3.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, LeakyReLU}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(LeakyReLU(inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.8846715	 -0.5720033	 -0.8220917
-0.51755846	 1.099684	 2.6011446
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.8846715	    -0.005720033   -0.008220917
-0.0051755845   1.099684	   2.6011446
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
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
[[0.14422043 0.38066946 0.55092494]
 [0.60075682 0.53505094 0.78330962]]
```
Output is
```python
[[0.14422044 0.38066944 0.55092496]
 [0.6007568  0.5350509  0.78330964]]
```

---
## **SReLU**
S-shaped Rectified Linear Unit.

It follows: f(x) = t^r + a^r(x - t^r) for x >= t^r, f(x) = x for t^r > x > t^l, f(x) = t^l + a^l(x - t^l) for x <= t^l.

**Scala:**
```scala
SReLU(tLeftInit = "zero", aLeftInit = "glorot_uniform", tRightInit = "glorot_uniform", aRightInit = "one", sharedAxes = null, inputShape = null)
```
**Python:**
```python
SReLU(t_left_init="zero", a_left_init="glorot_uniform", t_right_init="glorot_uniform", a_right_init="one", shared_axes=None, input_shape=None, name=None)
```

**Parameters:**

* `tLeftInit`: String representation of the initialization method for the left part intercept. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'zero'.
* `aLeftInit`: String representation of the initialization method for the left part slope. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'glorot_uniform'.
* `tRightInit`: String representation of ithe nitialization method for the right part intercept. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'glorot_uniform'.
* `aRightInit`: String representation of the initialization method for the right part slope. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'one'.
* `sharedAxes`: The axes along which to share learnable parameters for the activation function. Default is null.
For example, if the incoming feature maps are from a 2D convolution with output shape (batch, height, width, channels), and you wish to share parameters across space so that each filter only has one set of parameters, set 'sharedAxes = Array(1,2)'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, SReLU}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
model.add(SReLU(input_shape=(2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.42998132 0.47736492 0.9554154 ]
  [0.93264942 0.56851545 0.39508313]]

 [[0.5164102  0.22304862 0.44380779]
  [0.69137804 0.26413953 0.60638032]]]
```
Output is
```python
[[[0.42998132 0.47736493 0.9554154 ]
  [0.93264943 0.5685154  0.39508313]]

 [[0.5164102  0.22304863 0.44380778]
  [0.69137806 0.26413953 0.60638034]]]
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
ThresholdedReLU(theta=1.0, input_shape=None, name=None)
```

**Parameters:**

* `theta`: Threshold location of activation. Default is 1.0.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, ThresholdedReLU}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(ThresholdedReLU(inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
2.220999	1.2022058	-1.0015608
0.6532913	0.31831574	1.6747104
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
2.220999	1.2022058	0.0
0.0	        0.0	        1.6747104
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
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
[[0.0   0.0   0.0]
 [0.0   0.0   0.0]]
```
