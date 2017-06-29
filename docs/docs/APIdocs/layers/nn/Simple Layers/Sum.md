## Sum ##

**Scala:**
```scala
val m = Sum(dimension=1,nInputDims=-1,sizeAverage=false,squeeze=true)
```
**Python:**
```python
m = Sum(dimension=1,n_input_dims=-1,size_average=False)
```

Sum is a module that simply applies a sum operation over the given dimension - specified by the argument `dimension` (starting from 1). 
 
The input is expected to be either one tensor, or a batch of tensors (in mini-batch processing). If the input is a batch of tensors, you need to specify the number of dimensions of each tensor in the batch using `nInputDims`.  When input is one tensor, do not specify `nInputDims` or set it = -1, otherwise input will be interpreted as batch of tensors. 

**Scala example:**
```scala

scala> 
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val input = Tensor(2, 2, 2).randn()
val m1 = Sum(2)
val output1 = m1.forward(input)
val m2 = Sum(2, 1, true)
val output2 = m2.forward(input)

scala> print(input)
(1,.,.) =
-0.003314678    0.96401167
0.79000163      0.78624517

(2,.,.) =
-0.29975495     0.24742787
0.8709072       0.4381108

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x2]

scala> print(output1)
0.78668696      1.7502568
0.5711522       0.68553865
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

scala> print(output2)
0.39334348      0.8751284
0.2855761       0.34276932
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input=np.random.rand(2,2,2)
print "input is :",input
module = Sum(2)
out = module.forward(input)
print "output 1 is :",out
module = Sum(2,1,True)
out = module.forward(input)
print "output 2 is :",out
```
produces output:
```python
input is : [[[ 0.7194801   0.99120677]
  [ 0.07446639  0.056318  ]]

 [[ 0.08639016  0.17173268]
  [ 0.71686986  0.30503663]]]
creating: createSum
output 1 is : [array([[ 0.7939465 ,  1.04752481],
       [ 0.80325997,  0.47676933]], dtype=float32)]
creating: createSum
output 2 is : [array([[ 0.39697325,  0.5237624 ],
       [ 0.40162998,  0.23838466]], dtype=float32)]
```