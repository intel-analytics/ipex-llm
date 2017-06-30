## MM ##

**Scala:**
```scala
val m = MM(transA=false,transB=false)
```
**Python:**
```python
m = MM(trans_a=False,trans_b=False)
```


MM is a module that performs matrix multiplication on two mini-batch inputs, producing one mini-batch.

**Scala example:**
```scala
scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T

val input = T(1 -> Tensor(3, 3).randn(), 2 -> Tensor(3, 3).randn())
val m1 = MM()
val output1 = m1.forward(input)
val m2 = MM(true,true)
val output2 = m2.forward(input)

scala> print(input)
 {
        2: -0.62020904  -0.18690863     0.34132162
           -0.5359324   -0.09937895     0.86147165
           -2.6607985   -1.426654       2.3428898
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]
        1: -1.3087689   0.048720464     0.69583243
           -0.52055264  -1.5275089      -1.1569321
           0.28093573   -0.29353273     -0.9505267
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]
 }

scala> print(output1)
-1.0658705      -0.7529337      1.225519
4.2198563       1.8996398       -4.204146
2.512235        1.3327343       -2.38396
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

scala> print(output2)
1.0048954       0.99516183      4.8832207
0.15509865      -0.12717877     1.3618765
-0.5397563      -1.0767963      -2.4279075
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input1=np.random.rand(3,3)
input2=np.random.rand(3,3)
input = [input1,input2]
print "input is :",input
out = MM().forward(input)
print "output is :",out
```
produces output:
```python
input is : [array([[ 0.13696046,  0.92653165,  0.73585328],
       [ 0.28167852,  0.06431783,  0.15710073],
       [ 0.21896166,  0.00780161,  0.25780671]]), array([[ 0.11232797,  0.17023931,  0.92430042],
       [ 0.86629537,  0.07630215,  0.08584417],
       [ 0.47087278,  0.22992833,  0.59257503]])]
creating: createMM
output is : [array([[ 1.16452789,  0.26320592,  0.64217824],
       [ 0.16133308,  0.08898225,  0.35897085],
       [ 0.15274818,  0.09714822,  0.3558259 ]], dtype=float32)]
```