## Mean ##

**Scala:**
```scala
val m = Mean[T](dimension, nInputDims, squeeze)
```
**Python:**
```python
m = Mean(dimension=1,n_input_dims=-1)
```

Mean is a module that simply applies a mean operation over the given dimension - specified by `dimension` (starting from 1).

 
The input is expected to be either one tensor, or a batch of tensors (in mini-batch processing). If the input is a batch of tensors, you need to specify the number of dimensions of each tensor in the batch using `nInputDims`.  When input is one tensor, do not specify `nInputDims` or set it = -1, otherwise input will be interpreted as batch of tensors. 

**Scala example:**
```scala
scala> val input = Tensor[Double](2, 2, 2).randn()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,.,.) =
-0.5626799108296029     0.957065578369325
-0.5762423042889069     -1.9603044731082844

(2,.,.) =
0.779739764009551       -1.4229048510049056
-2.234790207521405      0.8517850653822254

[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2x2x2]

scala> val m1 = Mean[Double]()
m1: com.intel.analytics.bigdl.nn.Mean[Double] = nn.Mean

scala> m1.forward(input)
res16: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.10852992658997407     -0.23291963631779028
-1.4055162559051557     -0.5542597038630295
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

scala> val gradOutput = Tensor[Double](1,2,2).randn()
gradOutput: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,.,.) =
-0.570538239463982      1.608700755043723
-1.961741280513029      0.56076196585716

[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 1x2x2]

scala> m1.backward(input,gradOutput)
res17: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,.,.) =
-0.285269119731991      0.8043503775218614
-0.9808706402565145     0.28038098292858

(2,.,.) =
-0.285269119731991      0.8043503775218614
-0.9808706402565145     0.28038098292858

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2]

scala> val m2 = Mean[Double](2,1,true)
m2: com.intel.analytics.bigdl.nn.Mean[Double] = nn.Mean

scala> m2.forward(input)
res15: com.intel.analytics.bigdl.tensor.Tensor[Double] =
-0.5694611075592548     -0.5016194473694797
-0.7275252217559269     -0.2855598928113401
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
grad_out = np.random.rand(1,2,2)
grad_in = m1.backward(input,grad_out)
print "grad input of m1 is :",grad_in

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
grad input of m1 is : [array([[[ 0.29744586,  0.06938463],
        [ 0.44958934,  0.29279572]],

       [[ 0.29744586,  0.06938463],
        [ 0.44958934,  0.29279572]]], dtype=float32)]
creating: createMean
output m2 is : [array([[ 0.34840336,  0.527282  ],
       [ 0.39545399,  0.72673225]], dtype=float32)]
```