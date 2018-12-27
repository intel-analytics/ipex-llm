---
## **MaxPooling3D**
Applies max pooling operation for 3D data (spatial or spatio-temporal).

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

Border mode currently supported for this layer is 'valid'.

The input of this layer should be 5D.

**Scala:**
```scala
MaxPooling3D(poolSize = (2, 2, 2), strides = null, dimOrdering = "th", inputShape = null)
```
**Python:**
```python
MaxPooling3D(pool_size=(2, 2, 2), strides=None, border_mode="valid", dim_ordering="th", input_shape=None, name=None)
```

Parameters:

* `poolSize`: Length 3. Factors by which to downscale (dim1, dim2, dim3). Default is (2, 2, 2), which will halve the image in each dimension.
* `strides`: Length 3. Stride values. Default is null, and in this case it will be equal to poolSize.
* `dimOrdering`: Format of input data. Only 'th' (Channel First) is supported for now.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.MaxPooling3D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(MaxPooling3D(inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
-0.5052603	0.8938585	0.44785392
-0.48919395	0.35026026	0.541859

(1,1,2,.,.) =
1.5306468	0.24512683	1.71524
-0.49025944	2.1886358	0.15880944

(1,2,1,.,.) =
-0.5133986	-0.16549884	-0.2971134
1.5887301	1.8269571	1.3843931

(1,2,2,.,.) =
0.07515256	1.6993935	-0.3392596
1.2611006	0.20215735	1.3105171

(2,1,1,.,.) =
-2.0070438	0.35554957	0.21326075
-0.4078646	-1.5748956	-1.1007504

(2,1,2,.,.) =
1.0571382	-1.6031493	1.4638771
-0.25891435	1.4923956	-0.24045596

(2,2,1,.,.) =
-0.57790893	0.14577095	1.3165486
0.81937057	-0.3797079	1.2544848

(2,2,2,.,.) =
-0.42183575	-0.63774794	-2.0576336
0.43662143	1.9010457	-0.061519064

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
2.1886358

(1,2,1,.,.) =
1.8269571

(2,1,1,.,.) =
1.4923956

(2,2,1,.,.) =
1.9010457

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x1x1x1]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import MaxPooling3D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(MaxPooling3D(input_shape = (2, 2, 2, 3)))
input = np.random.random([2, 2, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[[[0.73349746 0.9811588  0.86071417]
    [0.33287621 0.37991739 0.87029317]]
   [[0.62537904 0.48099174 0.06194759]
    [0.38747972 0.05175308 0.36096032]]]
  [[[0.63260385 0.69990236 0.63353249]
    [0.19081261 0.56210617 0.75985185]]
   [[0.8624058  0.47224318 0.26524027]
    [0.75317792 0.39251436 0.98938982]]]]

 [[[[0.00556086 0.18833728 0.80340438]
    [0.9317538  0.88142596 0.90724509]]
   [[0.90243612 0.04594116 0.43662143]
    [0.24205094 0.58687822 0.57977055]]]
  [[[0.17240398 0.18346483 0.02520754]
    [0.06968248 0.02442692 0.56078895]]
   [[0.69503427 0.09528588 0.46104647]
    [0.16752596 0.88175901 0.71032998]]]]]
```
Output is:
```python
[[[[[0.9811588]]]
  [[[0.8624058]]]]

 [[[[0.9317538]]]
  [[[0.881759 ]]]]]
```
