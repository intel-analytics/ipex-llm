## Input ##

**Scala:**
```scala
val input = Input()
```
**Python:**
```python
input = Input()
```

Input layer do nothing to the input tensors, just passing them through.
It is used as input to the [Graph container](Containers.md#graph) when the first layer of the graph container accepts multiple tensors as inputs.

Each input node of the graph container should accept one tensor as input. If you want a module
accepting multiple tensors as input, you should add some Input module before it and connect
the outputs of the Input nodes to it. Please see the example of the Graph document.

Please note that the return is not a layer but a Node containing input layer.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val module = Input()
val input = Tensor(3, 2).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.93366385      0.82551944
0.71642804      0.4798109
0.83710635      0.068483874
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]

module.element.forward(input)
 com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.93366385      0.82551944
0.71642804      0.4798109
0.83710635      0.068483874
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Input()
input = np.random.rand(3,2)
array([[ 0.7006678 ,  0.29719472],
       [ 0.76668255,  0.59518023],
       [ 0.65543809,  0.41172803]])

module.element().forward(input)
array([[ 0.7006678 ,  0.29719472],
       [ 0.76668257,  0.59518021],
       [ 0.65543807,  0.41172802]], dtype=float32)

```

## Echo ##

**Scala:**
```scala
val module = Echo()
```
**Python:**
```python
module = Echo()
```

This module is for debug purpose, which can print activation and gradient size in your model topology

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Echo
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val module = Echo()
val input = Tensor(3, 2).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.24058184      0.22737113
0.0028103297    0.18359558
0.80443156      0.07047854
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]

module.forward(input)
res13: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.24058184      0.22737113
0.0028103297    0.18359558
0.80443156      0.07047854
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]
```

**Python example:**
```python
module = Echo()
input = np.random.rand(3,2)
[array([
[ 0.87273163,  0.59974301],
[ 0.09416127,  0.135765  ],
[ 0.11577505,  0.46095625]], dtype=float32)]

module.forward(input)
com.intel.analytics.bigdl.nn.Echo@535c681 : Activation size is 3x2
[array([
[ 0.87273163,  0.59974301],
[ 0.09416127,  0.135765  ],
[ 0.11577505,  0.46095625]], dtype=float32)]

```

