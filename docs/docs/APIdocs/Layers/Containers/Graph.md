## Graph ##

**Scala:**
```scala
val graph = Graph(Array(Node), Array(Node))
```
**Python:**
```python
model = Model([Node], [Node])
```

 A graph container. Each node can have multiple inputs. The output of the node should be a tensor.
 The output tensor can be connected to multiple nodes. So the module in each node can have a
 tensor or table input, and should have a tensor output.
 
 The graph container can have multiple inputs and multiple outputs. If there's one input, the
 input data fed to the graph module should be a tensor. If there're multiple inputs, the input
 data fed to the graph module should be a table, which is actually an sequence of tensor. The
 order of the input tensors should be same with the order of the input nodes. This is also
 applied to the gradient from the module in the back propagation.
 
 All of the input modules must accept a tensor input. If your input module accept multiple
 tensors as input, you should add some [Input layer](APIdocs/Layers/Utilities/Input.md) before
 it as input nodes and connect the output of the Input modules to that module.
 
 If there's one output, the module output is a tensor. If there're multiple outputs, the module
 output is a table, which is actually an sequence of tensor. The order of the output tensors is
 same with the order of the output modules. This is also applied to the gradient passed to the
 module in the back propagation.
 
 All inputs should be able to connect to outputs through some paths in the graph. It is
 allowed that some successors of the inputs node are not connect to outputs. If so, these nodes
 will be excluded in the computation.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat


val input1 = Input()
val input2 = Input()
val cadd = CAddTable().inputs(input1, input2)
val graph = Graph(Array(input1, input2), cadd)

val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
    Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
val gradInput = graph.backward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
    Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))),
    Tensor(T(0.1f, 0.2f, 0.3f, 0.4f)))

> println(output)
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.6
0.6
-0.5
-0.5
[com.intel.analytics.bigdl.tensor.DenseTensor of size 4]

> println(gradInput)
gradInput: com.intel.analytics.bigdl.nn.abstractnn.Activity =
 {
        2: 0.1
           0.2
           0.3
           0.4
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 4]
        1: 0.1
           0.2
           0.3
           0.4
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 4]
 }



```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np


input1 = Input()
input2 = Input()
cadd = CAddTable()([input1, input2])
model = Model([input1, input2], [cadd])
output = model.forward([
    np.array([0.1, 0.2, -0.3, -0.4]),
    np.array([0.5, 0.4, -0.2, -0.1])])

> output
array([ 0.60000002,  0.60000002, -0.5       , -0.5       ], dtype=float32)

gradInput = model.backward([
        np.array([0.1, 0.2, -0.3, -0.4]),
        np.array([0.5, 0.4, -0.2, -0.1])
    ],
    np.array([0.1, 0.2, 0.3, 0.4])
)

> gradInput
[array([ 0.1       ,  0.2       ,  0.30000001,  0.40000001], dtype=float32),
    array([ 0.1       ,  0.2       ,  0.30000001,  0.40000001], dtype=float32)]


```
