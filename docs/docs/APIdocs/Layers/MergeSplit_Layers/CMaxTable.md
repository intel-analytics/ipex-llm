## CMaxTable ##

**Scala:**
```scala
val m = CMaxTable()
```
**Python:**
```python
m = CMaxTable()
```

CMaxTable is a module that takes a table of Tensors and outputs the max of all of them.


**Scala example:**
```scala

scala> 
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T

val input1 = Tensor(3).randn()
val input2 =  Tensor(3).randn()
val input = T(input1, input2)
val m = CMaxTable()
val output = m.forward(input)
val gradOut = Tensor(3).randn()
val gradIn = m.backward(input,gradOut)

scala> print(input)
 {
        2: -0.38613814
           0.74074316
           -1.753783
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
        1: -1.6037064
           -2.3297918
           -0.7160026
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
 }

scala> print(output)
-0.38613814
0.74074316
-0.7160026
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]

scala> print(gradOut)
-1.4526331
0.7070323
0.29294914
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]

scala> print(gradIn)
 {
        2: -1.4526331
           0.7070323
           0.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
        1: 0.0
           0.0
           0.29294914
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
 }

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input1 = np.random.rand(3)
input2 = np.random.rand(3)
print "input is :",input1,input2

m = CMaxTable()
out = m.forward([input1,input2])
print "output of m is :",out

grad_out = np.random.rand(3)
grad_in = m.backward([input1, input2],grad_out)
print "grad input of m is :",grad_in
```
produces output:
```python
input is : [ 0.48649797  0.22131348  0.45667796] [ 0.73207053  0.74290136  0.03169769]
creating: createCMaxTable
output of m is : [array([ 0.73207051,  0.74290138,  0.45667794], dtype=float32)]
grad input of m is : [array([ 0.        ,  0.        ,  0.86938971], dtype=float32), array([ 0.04140199,  0.4787094 ,  0.        ], dtype=float32)]
```
