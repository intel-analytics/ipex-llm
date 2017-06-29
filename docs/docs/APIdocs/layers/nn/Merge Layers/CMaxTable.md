## CMaxTable ##

**Scala:**
```scala
val m = CMaxTable[T]()
```
**Python:**
```python
m = CMaxTable()
```

CMaxTable is a module that takes a table of Tensors and outputs the max of all of them.


**Scala example:**
```scala
scala>  val input1 = Tensor[Double](3).randn()
input1: com.intel.analytics.bigdl.tensor.Tensor[Double] =
-1.1299893907930936
-0.6668428117053015
-1.4320595866536823
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]

scala> val input2 =  Tensor[Double](3).randn()
input2: com.intel.analytics.bigdl.tensor.Tensor[Double] =
-0.10728239839248666
-0.3271036764589023
0.8346568193290994
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]

scala> import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.T

scala> val input = T(input1, input2)
input: com.intel.analytics.bigdl.utils.Table =
 {
        2: -0.10728239839248666
           -0.3271036764589023
           0.8346568193290994
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]
        1: -1.1299893907930936
           -0.6668428117053015
           -1.4320595866536823
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]
 }
 
scala> val m = CMaxTable[Double]()
m: com.intel.analytics.bigdl.nn.CMaxTable[Double] = CMaxTable[90ae67e9]

scala> m.forward(input)
res2: com.intel.analytics.bigdl.tensor.Tensor[Double] =
-0.10728239839248666
-0.3271036764589023
0.8346568193290994
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]

scala> val gradOut = Tensor[Double](3).randn()
gradOut: com.intel.analytics.bigdl.tensor.Tensor[Double] =
-1.0821492051923274
0.8070753324099773
-2.9081447772149924
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]

scala> m.backward(input,gradOut)
res3: com.intel.analytics.bigdl.utils.Table =
 {
        2: -1.0821492051923274
           0.8070753324099773
           -2.9081447772149924
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
        1: 0.0
           0.0
           0.0
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
