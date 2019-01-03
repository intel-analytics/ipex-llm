## **KerasLayerWrapper**
Wrap a torch style layer to keras style layer.

This layer can be built multiple times.

**Scala:**
```scala
KerasLayerWrapper(torchLayer, inputShape = null)
```
**Python:**
```python
KerasLayerWrapper(torch_layer, input_shape=None)
```

**Parameters:**

* `torchLayer`: a torch style layer.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.KerasLayerWrapper
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.nn.Linear
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
val dense = new KerasLayerWrapper[Float](Linear[Float](20, 10), inputShape = Shape(20))
model.add(dense)
val input = Tensor[Float](2, 20).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.55278283      -0.5434559      -0.13098523     0.3069534       -0.12007129     0.031956512     -0.019634819    -0.09178751     -1.2957728      1.3516346      1.3507701       -0.93318635     -1.1111038      1.0057137       0.093072094     0.16315712      -0.18079235     0.80998576      0.6703253     0.21223836
-1.007659       1.5507021       -0.14909777     0.49734116      1.4081444       0.1438721       1.7318599       -1.3321369      -0.6123855      0.43861434     0.9198252       1.1758715       -0.5824179      -0.90594006     -0.33974242     -0.58157283     1.3687168       -2.160458       -0.18854974   0.4541929
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x20]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.5819317       0.7231704       0.21700777      -0.1763548      0.02167879      0.19229038      0.7264892       -0.7566038      -0.8883222      0.47539598
-0.92322034     -0.33127156     0.48748493      -0.7715719      1.0859711       0.5226875       -0.6108173      -0.29417562     0.75702786      0.009688854
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x10]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import KerasLayerWrapper
from zoo.pipeline.api.keras.models import Sequential
from bigdl.nn.layer import Linear

model = Sequential()
model.add(KerasLayerWrapper(Linear(20, 10, with_bias=True) , input_shape=(20, )))
input = np.random.random([2, 20])
output = model.forward(input)
```
Input is:
```python
[[0.64178322, 0.83031778, 0.67272342, 0.3648695 , 0.37011444,
  0.87917395, 0.89792049, 0.93706952, 0.14721198, 0.76431214,
  0.11406789, 0.63280433, 0.72859274, 0.16546726, 0.94027721,
  0.7184913 , 0.04049882, 0.13775462, 0.88335614, 0.01030057],
 [0.69802784, 0.41952477, 0.79192261, 0.62655966, 0.00229703,
  0.74951992, 0.71846465, 0.72513163, 0.141432  , 0.54936796,
  0.18440429, 0.83081221, 0.42115396, 0.35078732, 0.35471522,
  0.2179049 , 0.95257499, 0.64030687, 0.95059945, 0.31188082]]
```
Output is
```python
[[-0.0319711 , -0.5341565 , -0.11790018, -0.7164225 , -0.10448539,
   0.03494176, -0.66940045,  0.6229225 ,  0.38492152, -0.527405  ],
 [-0.36529738, -0.57997525,  0.08127502, -0.7578952 , -0.1762895 ,
  -0.10188193, -0.18423618,  0.37726521,  0.21360731, -0.5451691 ]]
```
