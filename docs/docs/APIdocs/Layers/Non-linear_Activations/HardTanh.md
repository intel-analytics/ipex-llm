## HardTanh

**Scala:**
```scala
val activation = HardTanh(
    minValue = -1,
    maxValue = 1,
    inplace = false)
```
**Python:**
```python
activation = HardTanh(
    min_value=-1.0,
    max_value=1.0,
    inplace=False)
```

Applies non-linear function HardTanh to each element of input, HardTanh is defined:
```
           ⎧  maxValue, if x > maxValue
    f(x) = ⎨  minValue, if x < minValue
           ⎩  x, otherwise
```

**Parameters:**

**minValue** - minValue in f(x), default is -1.

**maxValue** - maxValue in f(x), default is 1.

**inplace**  - weather inplace update output from input. default is false.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.HardTanh
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val activation = HardTanh()
val input = Tensor(T(
  T(-1f, 2f, 3f),
  T(-2f, 3f, 4f),
  T(-3f, 4f, 5f)
))

val gradOutput = Tensor(T(
  T(3f, 4f, 5f),
  T(2f, 3f, 4f),
  T(1f, 2f, 3f)
))

val output = activation.forward(input)
val grad = activation.backward(input, gradOutput)

println(output)
-1.0	1.0	1.0
-1.0	1.0	1.0
-1.0	1.0	1.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

println(grad)
0.0	0.0	0.0
0.0	0.0	0.0
0.0	0.0	0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**
```python
activation = HardTanh()
input = np.array([
  [-1.0, 2.0, 3.0],
  [-2.0, 3.0, 4.0],
  [-3.0, 4.0, 5.0]
])

gradOutput = np.array([
  [3.0, 4.0, 5.0],
  [2.0, 3.0, 4.0],
  [1.0, 2.0, 5.0]
])

output = activation.forward(input)
grad = activation.backward(input, gradOutput)

print output
[[-1.  1.  1.]
 [-1.  1.  1.]
 [-1.  1.  1.]]

print grad
[[ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]]
```
