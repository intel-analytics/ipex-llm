## **Dropout**
Applies Dropout to the input by randomly setting a fraction 'p' of input units to 0 at each update during training time in order to prevent overfitting.

**Scala:**
```scala
Dropout(p, inputShape = null)
```
**Python:**
```python
Dropout(p, input_shape=None, name=None)
```

**Parameters:**

* `p`: Fraction of the input units to drop. Between 0 and 1.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.Dropout
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Dropout[Float](0.3, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.5496527       0.34846303      1.8184849       -0.8750735
-0.2907603      0.124056354     -0.5447822      -0.34512782
1.003834        -0.27847317     -0.16524693     -0.12172801

(2,.,.) =
-0.50297844     -0.78188837     -1.5617784      -1.2353797
-1.5052266      -1.6246556      0.5203618       1.144502
-0.18044183     -0.032648038    -1.9599762      -0.6970337

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
2.2137897       0.49780434      2.5978355       -1.250105
0.0     0.17722337      0.0     0.0
1.4340487       0.0     0.0     -0.17389716

(2,.,.) =
-0.71854067     -1.1169834      -2.231112       -1.7648282
-2.1503239      -2.3209367      0.743374        1.635003
-0.25777406     0.0     -2.799966       -0.99576247

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Dropout
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Dropout(0.3, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
array([[[0.80667346, 0.5370812 , 0.59039134, 0.48676815],
        [0.70987046, 0.11246564, 0.68062359, 0.48074257],
        [0.61979472, 0.36682032, 0.08320745, 0.41117697]],

       [[0.19616717, 0.18093539, 0.52080897, 0.73326568],
        [0.72752776, 0.81963229, 0.05652756, 0.37253947],
        [0.70200807, 0.27836313, 0.24421078, 0.58191582]]])
```
Output is
```python
array([[[1.1523907 , 0.7672588 , 0.        , 0.6953831 ],
        [1.0141007 , 0.1606652 , 0.9723194 , 0.6867751 ],
        [0.        , 0.5240291 , 0.11886779, 0.58739567]],

       [[0.2802388 , 0.        , 0.74401283, 1.0475224 ],
        [1.0393254 , 1.1709033 , 0.08075366, 0.53219926],
        [1.0028687 , 0.39766163, 0.        , 0.8313083 ]]], dtype=float32)
```