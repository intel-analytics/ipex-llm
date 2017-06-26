## CosineEmbeddingCriterion ##

**Scala:**
```scala
val cosineEmbeddingCriterion = CosineEmbeddingCriterion(val margin: Double = 0.0, val sizeAverage: Boolean = true)
```
**Python:**
```python
cosineEmbeddingCriterion = CosineEmbeddingCriterion( margin=0.0,size_average=True)
```
CosineEmbeddingCriterion creates a criterion that measures the loss given an input x = {x1, x2},
a table of two Tensors, and a Tensor label y with values 1 or -1.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val cosineEmbeddingCriterion = CosineEmbeddingCriterion(0.0, false)
val input1 = Tensor(5).rand()
val input2 = Tensor(5).rand()
val input = T()
input(1.0) = input1
input(2.0) = input2
val target1 = Tensor(Storage(Array(-0.5f)))
val target = T()
target(1.0) = target1

> print(cosineEmbeddingCriterion.forward(input, target))
0.7708353

> print(cosineEmbeddingCriterion.backward(input, target))
 {
	2: 0.22197613
	   0.3816089
	   0.113743804
	   0.020715533
	   -0.14027478
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
	1: -0.1818499
	   -0.31850398
	   0.083632
	   0.15572566
	   0.35671195
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
 }
```

**Python example:**
```python
from bigdl.nn.layer import *
cosineEmbeddingCriterion = CosineEmbeddingCriterion(0.0, False)
> cosineEmbeddingCriterion.forward([np.array([1.0, 2.0, 3.0, 4.0 ,5.0]),np.array([5.0, 4.0, 3.0, 2.0, 1.0])],[np.array(-0.5)])
0.6363636
> cosineEmbeddingCriterion.backward([np.array([1.0, 2.0, 3.0, 4.0 ,5.0]),np.array([5.0, 4.0, 3.0, 2.0, 1.0])],[np.array(-0.5)])
[array([ 0.07933884,  0.04958678,  0.01983471, -0.00991735, -0.03966942], dtype=float32), array([-0.03966942, -0.00991735,  0.01983471,  0.04958678,  0.07933884], dtype=float32)]

```

