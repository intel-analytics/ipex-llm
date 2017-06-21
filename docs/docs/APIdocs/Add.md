## Add ##

**Scala:**
```scala
val addLayer = Add[T](inputSize)
```
**Python:**
```python
add_layer = Add(input_size)
```

Description

A.K.A BiasAdd. This layer adds input tensor with a parameter tensor and output the result.
If the input is 1D, this layer just do a element-wise add. If the input has multiple dimentions,
this layer will treat the first dimension as batch dimension, resize the input tensor to a 2D 
tensor(batch-dimension x input_size) and do a broadcast add between the 2D tensor and the 
parameter.

Please note that the parameter will be trained in the back propagation.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor

val addLayer = Add[Float](4)
addLayer.bias.set(Tensor[Float](T(1.0f, 2.0f, 3.0f, 4.0f)))
addLayer.forward(Tensor[Float](T(T(1.0f, 1.0f, 1.0f, 1.0f), T(3.0f, 3.0f, 3.0f, 3.0f))))
addLayer.backward(Tensor[Float](T(T(1.0f, 1.0f, 1.0f, 1.0f), T(3.0f, 3.0f, 3.0f, 3.0f))),
    Tensor[Float](T(T(0.1f, 0.1f, 0.1f, 0.1f), T(0.3f, 0.3f, 0.3f, 0.3f))))
```
Its output should be
```
2.0     3.0     4.0     5.0
4.0     5.0     6.0     7.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]

0.1     0.1     0.1     0.1
0.3     0.3     0.3     0.3
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]
```

**Python example:**
```python
from bigdl.nn.layer import Add
import numpy as np

add_layer = Add(4)
add_layer.set_weights([np.array([1.0, 2.0, 3.0, 4.0])])
add_layer.forward(np.array([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]]))
add_layer.backward(np.array([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]]),
    np.array([[0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3]]))
```
Its output should be
```
array([[ 2.,  3.,  4.,  5.],
       [ 4.,  5.,  6.,  7.]], dtype=float32)
       
array([[ 0.1       ,  0.1       ,  0.1       ,  0.1       ],
       [ 0.30000001,  0.30000001,  0.30000001,  0.30000001]], dtype=float32)   
```