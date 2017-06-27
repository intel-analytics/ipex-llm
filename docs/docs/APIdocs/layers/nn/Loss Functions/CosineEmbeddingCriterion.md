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
impot com.intel.analytics.bigdl.utils.T
val cosineEmbeddingCriterion = CosineEmbeddingCriterion(0.0, false)
val input1 = Tensor(5).rand()
val input2 = Tensor(5).rand()
val input = T()
input(1.0) = input1
input(2.0) = input2
val target1 = Tensor(Storage(Array(-0.5f)))
val target = T()
target(1.0) = target1

> print(input)
 {
	2.0: 0.4110882
	     0.57726574
	     0.1949834
	     0.67670715
	     0.16984987
	     [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]
	1.0: 0.16878392
	     0.24124223
	     0.8964794
	     0.11156334
	     0.5101486
	     [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]
 }

> print(cosineEmbeddingCriterion.forward(input, target))
0.49919847

> print(cosineEmbeddingCriterion.backward(input, target))
 {
	2: -0.045381278
	   -0.059856333
	   0.72547954
	   -0.2268434
	   0.3842142
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
	1: 0.30369008
	   0.42463788
	   -0.20637506
	   0.5712836
	   -0.06355385
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

