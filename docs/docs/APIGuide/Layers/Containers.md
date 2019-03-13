## Sequential ##

**Scala:**
```scala
val module = Sequential()
```
**Python:**
```python
seq = Sequential()
```

Sequential provides a means to plug layers together
in a feed-forward fully connected manner.

**Scala example:**

```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

import com.intel.analytics.bigdl.nn.{Sequential, Linear}

val module = Sequential()
module.add(Linear(10, 25))
module.add(Linear(25, 10))

val input = Tensor(10).range(1, 10, 1)
val gradOutput = Tensor(10).range(1, 10, 1)

val output = module.forward(input).toTensor
val gradInput = module.backward(input, gradOutput).toTensor

println(output)
println(gradInput)
```
Gives the output,

```
-2.3750305
2.4512818
1.6998017
-0.47432393
4.3048754
-0.044168986
-1.1643536
0.60341483
2.0216258
2.1190155
[com.intel.analytics.bigdl.tensor.DenseTensor of size 10]
```
Gives the gradInput,

```
2.593382
-1.4137214
-1.8271983
1.229643
0.51384985
1.509845
2.9537349
1.088281
0.2618509
1.4840821
[com.intel.analytics.bigdl.tensor.DenseTensor of size 10]
```

**Python example:**

```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

seq = Sequential()
seq.add(Linear(10, 25))
seq.add(Linear(25, 10))

input = np.arange(1, 11, 1).astype("float32")
input = input.reshape(1, 10)

output = seq.forward(input)
print output

gradOutput = np.arange(1, 11, 1).astype("float32")
gradOutput = gradOutput.reshape(1, 10)

gradInput = seq.backward(input, gradOutput)
print gradInput
```
Gives the output,

```
[array([[ 1.08462083, -2.03257799, -0.5400058 ,  0.27452484,  1.85562158,
         1.64338267,  2.45694995,  1.70170391, -2.12998056, -1.28924525]], dtype=float32)]
```

Gives the gradInput,

```

[array([[ 1.72007763,  1.64403224,  2.52977395, -1.00021958,  0.1134415 ,
         2.06711197,  2.29631734, -3.39587498,  1.01093054, -0.54482007]], dtype=float32)]
```

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
 tensors as input, you should add some [Input layer](Utilities.md#input) before
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

## Concat ##

**Scala:**
```scala
val module = Concat(dimension)
```
**Python:**
```python
module = Concat(dimension)
```

Concat is a container who concatenates the output of it's submodules along the
provided `dimension`: all submodules take the same inputs, and their output is
concatenated.
```
                 +----Concat----+
            +---->  submodule1  -----+
            |    |              |    |
 input -----+---->  submodule2  -----+----> output
            |    |              |    |
            +---->  submodule3  -----+
                 +--------------+
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val mlp = Concat(2)
mlp.add(Linear(3,2))
mlp.add(Linear(3,4))

println(mlp.forward(Tensor(2, 3).rand()))
```
Gives the output,
```
-0.17087375	0.12954286	0.15685591	-0.027277306	0.38549712	-0.20375136
-0.9473443	0.030516684	0.23380546	0.625985	-0.031360716	0.40449825
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x6]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

mlp = Concat(2)
mlp.add(Linear(3,2))
mlp.add(Linear(3,4))
print(mlp.forward(np.array([[1, 2, 3], [1, 2, 3]])))
```
Gives the output,
```
[array([
[-0.71994132,  2.17439198, -1.46522939,  0.64588934,  2.61534023, -2.39528942],
[-0.89125222,  5.49583197, -2.8865242 ,  1.44914722,  5.26639175, -6.26586771]]
      dtype=float32)]

```

## ParallelTable ##

**Scala:**
```scala
val module = ParallelTable()
```
**Python:**
```python
module = ParallelTable()
```

It is a container module that applies the i-th member module to the i-th
 input, and outputs an output in the form of Table
 
```
+----------+         +-----------+
| {input1, +---------> {member1, |
|          |         |           |
|  input2, +--------->  member2, |
|          |         |           |
|  input3} +--------->  member3} |
+----------+         +-----------+

```
 
**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T 
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val module = ParallelTable()
val log = Log()
val exp = Exp()
module.add(log)
module.add(exp)
val input1 = Tensor(3, 3).rand(0, 1)
val input2 = Tensor(3).rand(0, 1)
val input = T(1 -> input1, 2 -> input2)
> print(module.forward(input))
 {
        2: 2.6996834
           2.0741253
           1.0625387
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
        1: -1.073425    -0.6672964      -1.8160943
           -0.54094607  -1.3029919      -1.7064717
           -0.66175103  -0.08288143     -1.1840979
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
 }
```

**Python example:**
```python
from bigdl.nn.layer import *

module = ParallelTable()
log = Log()
exp = Exp()
module.add(log)
module.add(exp)
input1 = np.random.rand(3,3)
input2 = np.random.rand(3)
>module.forward([input1, input2])
[array([[-1.27472472, -2.18216252, -0.60752904],
        [-2.76213861, -1.77966928, -0.13652121],
        [-1.47725129, -0.03578046, -1.37736678]], dtype=float32),
 array([ 1.10634041,  1.46384597,  1.96525407], dtype=float32)]
```

## ConcatTable ##

**Scala:**
```scala
val module = ConcatTable()
```
**Python:**
```python
module = ConcatTable()
```

ConcateTable is a container module like Concate. Applies an input
to each member module, input can be a tensor or a table.

ConcateTable usually works with CAddTable and CMulTable to
 implement element wise add/multiply on outputs of two modules.

```
                   +-----------+
             +----> {member1, |
+-------+    |    |           |
| input +----+---->  member2, |
+-------+    |    |           |
   or        +---->  member3} |
 {input}          +-----------+
 
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val mlp = ConcatTable()
mlp.add(Linear(3, 2))
mlp.add(Linear(3, 4))

> print(mlp.forward(Tensor(2, 3).rand()))

{
	2: -0.37111914	0.8542446	-0.362602	-0.75522065	
	   -0.28713673	0.6021913	-0.16525984	-0.44689763	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]
	1: -0.79941726	0.8303885	
	   -0.8342782	0.89961016	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
 }

```

**Python example:**
```python
from bigdl.nn.layer import *

mlp = ConcatTable()
mlp.add(Linear(3, 2))   
mlp.add(Linear(3, 4))
> mlp.forward(np.array([[1, 2, 3], [1, 2, 3]]))
out: [array([[ 1.16408789, -0.1508013 ],
             [ 1.16408789, -0.1508013 ]], dtype=float32),
      array([[-0.24672163, -0.56958938, -0.51390374,  0.64546645],
             [-0.24672163, -0.56958938, -0.51390374,  0.64546645]], dtype=float32)]

```

## Bottle ##

**Scala:**
```scala
val model = Bottle(module, nInputDim, nOutputDim)
```
**Python:**
```python
model = Bottle(module, nInputDim, nOutputDim)
```

Bottle allows varying dimensionality input to be forwarded through any module that accepts input of nInputDim dimensions, and generates output of nOutputDim dimensions.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val model = Bottle(Linear(3, 2).asInstanceOf[Module[Float]], 2, 2)
val input = Tensor(2, 3, 3).rand()

scala> print(input)
(1,.,.) =
0.7843752	0.17286697	0.20767091	
0.8594811	0.9100018	0.8448141	
0.7683892	0.36661968	0.76637685	

(2,.,.) =
0.7163263	0.083962396	0.81222403	
0.7947034	0.09976136	0.114404656	
0.14890474	0.43289232	0.1489096	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3x3]	

val output = model.forward(input)

scala> print(output)
(1,.,.) =
-0.31146684	0.40719786	
-0.51778656	0.58715886	
-0.51676923	0.4027511	

(2,.,.) =
-0.5498678	0.29658738	
-0.280177	0.39901164	
-0.2387946	0.24809375	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x2]
```

**Python example:**
```python
model = Bottle(Linear(3, 2), 2, 2)

input = np.random.randn(2, 3, 3)
output = model.forward(input)

>>> print(input)
[[[ 0.42370589 -1.7938942   0.56666373]
  [-1.78501381  0.55676471 -0.50150367]
  [-1.59262182  0.82079469  1.1873599 ]]

 [[ 0.95799792 -0.71447244  1.05344083]
  [-0.07838376 -0.88780484 -1.80491177]
  [ 0.99996222  1.39876002 -0.16326094]]]
>>> print(output)
[[[ 0.26298434  0.74947536]
  [-1.24375117 -0.33148435]
  [-1.35218966  0.17042145]]

 [[ 0.08041853  0.91245329]
  [-0.08317742 -0.13909879]
  [-0.52287608  0.3667658 ]]]
```

## MapTable ##

**Scala:**
```scala
val mod = MapTable(module=null)
```
**Python:**
```python
mod = MapTable(module=None)
```

This class is a container for a single module which will be applied
to all input elements. The member module is cloned as necessary to
process all input elements.

`module` a member module.  

```
+----------+         +-----------+
| {input1, +---------> {member,  |
|          |         |           |
|  input2, +--------->  clone,   |
|          |         |           |
|  input3} +--------->  clone}   |
+----------+         +-----------+
```
 
**Scala example:**

```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T 

val map = MapTable()
map.add(Linear(10, 3))
val input = T(
      Tensor(10).randn(),
      Tensor(10).randn())
> print(map.forward(input))
{
	2: 0.2444828
	   -1.1700082
	   0.15887381
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
	1: 0.06696482
	   0.18692614
	   -1.432079
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
 }
```

**Python example:**
```python
from bigdl.nn.layer import *

map = MapTable()
map.add(Linear(10, 3))
input = [np.random.rand(10), np.random.rand(10)]
>map.forward(input)
[array([ 0.69586945, -0.70547599, -0.05802459], dtype=float32),
 array([ 0.47995114, -0.67459631, -0.52500772], dtype=float32)]
```

## Container ##

Container is a subclass of abstract class AbstractModule, which
declares methods defined in all containers. A container usually
contains some other modules in the `modules` variable. It overrides
many module methods such that calls are propagated to the contained
modules.

