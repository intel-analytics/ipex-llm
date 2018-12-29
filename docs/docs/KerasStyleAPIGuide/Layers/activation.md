## **Activation**
Simple activation function to be applied to the output.

**Scala:**
```scala
Activation(activation, inputShape = null)
```
**Python:**
```python
Activation(activation, input_shape=None, name=None)
```

Parameters:

* `activation`: Name of the activation function as string. See [here](#available-activations) for available activation strings.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.Activation
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Activation[Float]("tanh", inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
2.1659365	0.28006053	-0.20148286
0.9146865	 3.4301455	  1.0930616
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.9740552	 0.2729611	  -0.1988
 0.723374	0.99790496	0.7979928
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Activation
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Activation("tanh", input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[ 0.26202468  0.15868397  0.27812652]
 [ 0.45931689  0.32100054  0.51839282]]
```
Output is
```python
[[ 0.2561883   0.15736534  0.27117023]
 [ 0.42952728  0.31041133  0.47645861]]
```

Note that the following two pieces of code will be equivalent:
```python
model.add(Dense(32))
model.add(Activation('relu'))
```
```python
model.add(Dense(32, activation="relu"))
```

---
## **HardTanh**
Applies the hard tanh function element-wise to the input.

f(x) = maxValue, if x > maxValue

f(x) = minValue, if x < minValue

f(x) = x, otherwise

When you use this layer as the first layer of a model, you need to provide the argument input_shape (a shape tuple, does not include the batch dimension).

Remark: This layer is from Torch and wrapped in Keras style.

**Scala:**
```scala
HardTanh(minValue = -1, maxValue = 1, inputShape = null)
```
**Python:**
```python
HardTanh(min_value=-1, max_value=1, input_shape=None, name=None)
```

**Parameters:**

* `minValue`: The minimum threshold value. Default is -1.
* `maxValue`: The maximum threshold value. Default is 1.
* `inputShape`: A Single Shape, does not include the batch dimension.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.HardTanh
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(HardTanh[Float](-1, 0.5, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.8396661       -2.096241       -0.36010137     -1.97987
-0.20326108     1.5972694       -1.4166505      -0.3369559
-0.22637285     -1.1021988      1.0707928       -1.5014135

(2,.,.) =
-0.24511681     -1.1103313      -0.7901563      -1.0394055
-0.033373486    0.22657289      -0.7928737      1.5241393
0.49224186      -0.21418595     -0.32379007     -0.941034

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.5     -1.0    -0.36010137     -1.0
-0.20326108     0.5     -1.0    -0.3369559
-0.22637285     -1.0    0.5     -1.0

(2,.,.) =
-0.24511681     -1.0    -0.7901563      -1.0
-0.033373486    0.22657289      -0.7928737      0.5
0.49224186      -0.21418595     -0.32379007     -0.941034

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import HardTanh
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(HardTanh(-1, 0.5, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.38707977, 0.94085094, 0.50552125, 0.42818523],
  [0.5544486 , 0.36521357, 0.42551631, 0.93228245],
  [0.29155494, 0.61710319, 0.93137551, 0.05688166]],

 [[0.75222706, 0.36454257, 0.83076327, 0.82004643],
  [0.29213453, 0.71532663, 0.99556398, 0.57001469],
  [0.58088671, 0.32646428, 0.60736   , 0.14861018]]]
```
Output is
```python
[[[0.38707978, 0.5       , 0.5       , 0.42818522],
  [0.5       , 0.36521357, 0.4255163 , 0.5       ],
  [0.29155496, 0.5       , 0.5       , 0.05688166]],

  [[0.5       , 0.36454257, 0.5       , 0.5       ],
   [0.29213452, 0.5       , 0.5       , 0.5       ],
   [0.5       , 0.3264643 , 0.5       , 0.14861017]]]
```
