## Cosine ##

**Scala:**
```scala
val m = Cosine[T](inputSize, outputSize)
```
**Python:**
```python
m = Cosine(input_size, output_size)
```

Cosine is a module used to  calculate the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) of the input to `outputSize` centers, i.e. this layer has the weights `w_j`, for `j = 1,..,outputSize`, where `w_j` are vectors of dimension `inputSize`.

The distance `y_j` between center `j` and input `x` is formulated as `y_j = (x · w_j) / ( || w_j || * || x || )`.

The input given in `forward(input)` must be either a vector (1D tensor) or matrix (2D tensor). If the input is a
vector, it must have the size of `inputSize`. If it is a matrix, then each row is assumed to be an input sample of given batch (the number of rows means the batch size and the number of columns should be equal to the `inputSize`).
	
**Scala example:**
```scala
scala> val input = Tensor[Double](3, 2).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.7678213366307318      0.27321007940918207
0.13674732483923435     0.05386950820684433
0.6806434956379235      0.827141807647422
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x2]

scala> val m = Cosine[Double](2, 3)
m: com.intel.analytics.bigdl.nn.Cosine[Double] = Cosine[168435a9](2, 3)

scala> m.forward(input)
res5: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.5517294422782427      0.4692623535624056      -0.8678327249076271
0.5235603322968813      0.4395011979673559      -0.8839462296536033
0.044021543454757596    -0.05192204006469456    -0.9997913572841641
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input=np.random.rand(2,3)
print "input is :",input
module = Cosine(3,3)
module.forward(input)
print "output is :",out
```

produces output:
```python
input is : [[ 0.31156943  0.85577626  0.4274042 ]
 [ 0.79744055  0.66431136  0.05657437]]
creating: createCosine
output is : [array([[-0.73284394, -0.28076306, -0.51965958],
       [-0.9563939 , -0.42036989, -0.08060561]], dtype=float32)]


```