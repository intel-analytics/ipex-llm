## **PReLU**
Applies parametric ReLU, where parameter varies the slope of the negative part.

It follows: f(x) = max(0, x) + a * min(0, x)

**Scala:**
```scala
PReLU(nOutputPlane = 0, inputShape = null)
```
**Python:**
```python
PReLU(nOutputPlane=0, input_shape=None)
```

**Parameters:**

* `nOutputPlane`: Input map number. Default is 0,
                  which means using PReLU in shared version and has only one parameter.
* `inputShape`:  A Single Shape, does not include the batch dimension.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.PReLU
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(PReLU[Float](inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.9026888      -1.0402212      1.3878769
-0.17167428     0.08202032      1.2682742
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
-0.2256722      -0.2600553      1.3878769
-0.04291857     0.08202032      1.2682742
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import PReLU

model = Sequential()
model.add(PReLU(input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.61639702 0.08877075 0.93652509]
 [0.38800821 0.76286851 0.95777973]]
```
Output is
```python
[[0.616397   0.08877075 0.9365251 ]
 [0.3880082  0.7628685  0.9577797 ]]
```

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
ELU(alpha=1.0, input_shape=None, name=None)
```

**Parameters:**

* `alpha`: Scale for the negative factor. Default is 1.0.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.ELU
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(ELU[Float](inputShape = Shape(3)))
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
from zoo.pipeline.api.keras.layers import ELU
from zoo.pipeline.api.keras.models import Sequential

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
import com.intel.analytics.zoo.pipeline.api.keras.layers.SReLU
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(SReLU[Float](inputShape = Shape(2, 3)))
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
from zoo.pipeline.api.keras.layers import SReLU
from zoo.pipeline.api.keras.models import Sequential

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
import com.intel.analytics.zoo.pipeline.api.keras.layers.ThresholdedReLU
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(ThresholdedReLU[Float](inputShape = Shape(3)))
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
from zoo.pipeline.api.keras.layers import ThresholdedReLU
from zoo.pipeline.api.keras.models import Sequential

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