## Mean ##

**Scala:**
```scala
val m = Mean(dimension=1, nInputDims=-1, squeeze=true)
```
**Python:**
```python
m = Mean(dimension=1,n_input_dims=-1, squeeze=True)
```

Mean is a module that simply applies a mean operation over the given dimension - specified by `dimension` (starting from 1).

 
The input is expected to be either one tensor, or a batch of tensors (in mini-batch processing). If the input is a batch of tensors, you need to specify the number of dimensions of each tensor in the batch using `nInputDims`.  When input is one tensor, do not specify `nInputDims` or set it = -1, otherwise input will be interpreted as batch of tensors. 

**Scala example:**
```scala
scala> 
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val input = Tensor(2, 2, 2).randn()
val m1 = Mean()
val output1 = m1.forward(input)
val m2 = Mean(2,1,true)
val output2 = m2.forward(input)

scala> print(input)
(1,.,.) =
-0.52021635     -1.8250599
-0.2321481      -2.5672712

(2,.,.) =
4.007425        -0.8705412
1.6506456       -0.2470611

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x2]

scala> print(output1)
1.7436042       -1.3478005
0.7092488       -1.4071661
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

scala> print(output2)
-0.37618223     -2.1961656
2.8290353       -0.5588012
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input = np.random.rand(2,2,2)
print "input is :",input

m1 = Mean()
out = m1.forward(input)
print "output m1 is :",out

m2 = Mean(2,1,True)
out = m2.forward(input)
print "output m2 is :",out
```
produces output:
```python
input is : [[[ 0.01990713  0.37740696]
  [ 0.67689963  0.67715705]]

 [[ 0.45685026  0.58995121]
  [ 0.33405769  0.86351324]]]
creating: createMean
output m1 is : [array([[ 0.23837869,  0.48367909],
       [ 0.50547862,  0.77033514]], dtype=float32)]
creating: createMean
output m2 is : [array([[ 0.34840336,  0.527282  ],
       [ 0.39545399,  0.72673225]], dtype=float32)]
```