## **SparseDense**
SparseDense is the sparse version of layer Dense. SparseDense has two different from Dense:
firstly, SparseDense's input Tensor is a SparseTensor. Secondly, SparseDense doesn't backward
gradient to next layer in the backpropagation by default, as the gradInput of SparseDense is
useless and very big in most cases.

But, considering model like Wide&Deep, we provide backwardStart and backwardLength to backward
part of the gradient to next layer.

The most common input is 2D.

When you use this layer as the first layer of a model, you need to provide the argument
inputShape (a Single Shape, does not include the batch dimension).

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
