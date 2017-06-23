## Sum ##

**Scala:**
```scala
val m = Sum[T](dimension,nInputDims,sizeAverage,squeeze)
```
**Python:**
```python
m = Sum(dimension,n_input_dims,size_average)
```

Sum is a module that simply applies a sum operation over the given dimension - specified by the argument `dimension` (starting from 1). 
 
The input is expected to be either one tensor, or a batch of tensors (in mini-batch processing). If the input is a batch of tensors, you need to specify the number of dimensions of each tensor in the batch using `nInputDims`.  When input is one tensor, do not specify `nInputDims` or set it = -1, otherwise input will be interpreted as batch of tensors. 

**Scala example:**
```scala
scala> val input = Tensor[Double](2, 2, 2).randn()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,.,.) =
1.5835003800372862      0.9237150600038898
0.5103501798991353      -0.6014806933883298

(2,.,.) =
-0.05068578108652042    0.8339699884635934
-1.7135159288916593     0.1351458541052799

[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2x2x2]

scala> val m = Sum[Double](2)
m: com.intel.analytics.bigdl.nn.Sum[Double] = nn.Sum

scala> m.forward(input)
res12: com.intel.analytics.bigdl.tensor.Tensor[Double] =
2.0938505599364214      0.32223436661556004
-1.7642017099781797     0.9691158425688733
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

scala> val m2 = Sum[Double](2, 1, true)
m2: com.intel.analytics.bigdl.nn.Sum[Double] = nn.Sum

scala> m2.forward(input)
res13: com.intel.analytics.bigdl.tensor.Tensor[Double] =
1.0469252799682107      0.16111718330778002
-0.8821008549890899     0.48455792128443664
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