## **SparseEmbedding**
SparseEmbedding is the sparse version of layer Embedding.

The input of SparseEmbedding should be a 2D SparseTensor or two 2D sparseTensors.
If the input is a SparseTensor, the values are positive integer ids,
values in each row of this SparseTensor will be turned into a dense vector.
If the input is two SparseTensors, the first tensor should be the integer ids, just
like the SparseTensor input. And the second tensor is the corresponding
weights of the integer ids.

This layer can only be used as the first layer in a model, you need to provide the argument
inputShape (a Single Shape, does not include the batch dimension).

**Scala:**
```scala
SparseEmbedding(inputDim, outputDim, combiner = "sum", max_norm = -1.0, init = "uniform", wRegularizer = null, inputShape = null)
```
**Python:**
```python
SparseEmbedding(input_dim, output_dim, combiner="sum", max_norm=-1.0, init="uniform", W_regularizer=None, input_shape=None, name=None)
```

**Parameters:**

* `inputDim`: Int > 0. Size of the vocabulary.
* `outputDim`: Int >= 0. Dimension of the dense embedding.
* `init`: String representation of the initialization method for the weights of the layer. Default is "uniform".
* `combiner`: A string specifying the reduce type.
              Currently "mean", "sum", "sqrtn" is supported.
* `maxNorm`: If provided, each embedding is normalized to have l2 norm equal to
               maxNorm before combining.
* `wRegularizer`: An instance of [Regularizer], (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a `Shape` object. For Python API, it should be a shape tuple. Batch dimension should be excluded.
* `name`: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.SparseEmbedding
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val indices1 = Array(0, 0, 1, 2)
val indices2 = Array(0, 1, 0, 3)
val values = Array(2f, 4, 1, 2)
val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))
val layer = SparseEmbedding[Float](inputDim = 10, outputDim = 4,
  combiner = "sum", inputShape = Shape(10))
layer.build(Shape(-1, 10))
val output = layer.forward(input)
```
Input is:
```scala
input: 
(0, 0) : 2.0
(0, 1) : 4.0
(1, 0) : 1.0
(2, 3) : 2.0
[com.intel.analytics.bigdl.tensor.SparseTensor of size 3x4]
```
Output is:
```scala
-0.03674142	-0.01844017	-0.015794257	-0.045957662	
-0.02645839	-0.024193227	-0.046255145	-0.047514524	
-0.042759597	0.002117775	-0.041510757	1.9092667E-4	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import SparseEmbedding
from zoo.pipeline.api.keras.models import Sequential
from bigdl.util.common import JTensor

model = Sequential()
model.add(SparseEmbedding(input_dim=10, output_dim=4, input_shape=(4, )))
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
[[ 0.00771878 -0.05676365  0.03861053  0.04300173]
 [-0.04647886 -0.03346863  0.04642192 -0.0145219 ]
 [ 0.03964841  0.0243053   0.04841208  0.04862341]]
```
