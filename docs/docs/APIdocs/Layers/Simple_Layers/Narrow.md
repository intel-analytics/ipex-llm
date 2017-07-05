## Narrow

**Scala:**
```scala
val layer = Narrow(dimension, offset, length = 1)
```
**Python:**
```python
layer = Narrow(dimension, offset, length=1)
```

Narrow is an application of narrow operation in a module.
The module further supports a negative length in order to handle inputs with an unknown size.

**Parameters:**

**dimension** -narrow along this dimension

**offset** -the start index on the given dimension

**length** -length to narrow, default value is 1

**Scala Example**
```scala
import com.intel.analytics.bigdl.nn.Narrow
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val layer = Narrow(2, 2)
val input = Tensor(T(
  T(-1f, 2f, 3f),
  T(-2f, 3f, 4f),
  T(-3f, 4f, 5f)
))

val gradOutput = Tensor(T(3f, 4f, 5f))

val output = layer.forward(input)
2.0
3.0
4.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x1]

val grad = layer.backward(input, gradOutput)
0.0	3.0	0.0
0.0	4.0	0.0
0.0	5.0	0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python Example**
```python
layer = Narrow(2, 2)
input = np.array([
  [-1.0, 2.0, 3.0],
  [-2.0, 3.0, 4.0],
  [-3.0, 4.0, 5.0]
])

gradOutput = np.array([3.0, 4.0, 5.0])

output = layer.forward(input)
grad = layer.backward(input, gradOutput)

print output
[[ 2.]
 [ 3.]
 [ 4.]]

print grad
[[ 0.  3.  0.]
 [ 0.  4.  0.]
 [ 0.  5.  0.]]
```