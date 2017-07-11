## Cosine ##

**Scala:**
```scala
val m = Cosine(inputSize, outputSize)
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
scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val m = Cosine(2, 3)
val input = Tensor(3, 2).rand()
val output = m.forward(input)

scala> print(input)
0.48958543      0.38529378
0.28814933      0.66979927
0.3581584       0.67365724
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]

scala> print(output)
0.998335        0.9098057       -0.71862763
0.8496431       0.99756527      -0.2993874
0.8901594       0.9999207       -0.37689084
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