## Index ##

**Scala:**
```scala
val model = Index[T](dimension)
```
**Python:**
```python
model = Index(dimension)
```

Applies the Tensor index operation along the given dimension.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

val input1 = Tensor[Float](3).rand()
val input2 = Tensor[Float](4)
input2(Array(1)) = 1
input2(Array(2)) = 2
input2(Array(3)) = 2
input2(Array(4)) = 3

val input = T(input1, input2)
val model = Index[Float](1)
val output = model.forward(input)
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
0.6175723
0.4498806
0.4498806
0.41750473
[com.intel.analytics.bigdl.tensor.DenseTensor of size 4]
```

**Python example:**
```python
input1 = np.random.randn(3)
input2 = np.array([1, 2, 2, 3])
input = [input1, input2]

model = Index(1)
output = model.forward(input)
```
output is
```
array([-0.34750494,  0.31201595,  0.31201595,  0.96357429], dtype=float32)
```
