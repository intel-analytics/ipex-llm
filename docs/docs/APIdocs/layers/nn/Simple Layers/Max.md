## Max

**Scala:**
```scala
val layer = Max(dim = 1, numInputDims = Int.MinValue)
```
**Python:**
```python
layer = Max(dim, num_input_dims=INTMIN)
```

Applies a max operation over dimension `dim`.

**Parameters:**

**dim** max along this dimension

**numInputDims** Optional. If in a batch model, set to the inputDims.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Max
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val layer = Max(1, 1)
val input = Tensor(T(
  T(-1f, 2f, 3f),
  T(-2f, 3f, 4f),
  T(-3f, 4f, 5f)
))

val gradOutput = Tensor(T(3f, 4f, 5f))

val output = layer.forward(input)
val grad = layer.backward(input, gradOutput)

println(output)
3.0
4.0
5.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]

println(grad)
0.0	0.0	3.0
0.0	0.0	4.0
0.0	0.0	5.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**
```python
layer = Max(1, 1)
input = np.array([
  [-1.0, 2.0, 3.0],
  [-2.0, 3.0, 4.0],
  [-3.0, 4.0, 5.0]
])

gradOutput = np.array([3.0, 4.0, 5.0])

output = layer.forward(input)
grad = layer.backward(input, gradOutput)

print output
[ 3.  4.  5.]

print grad
[[ 0.  0.  3.]
 [ 0.  0.  4.]
 [ 0.  0.  5.]]
``