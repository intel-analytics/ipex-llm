## SoftShrink
**Scala:**
```scala
val layer = SoftShrink(lambda = 0.5)
```
**Python:**
```python
layer = SoftShrink(the_lambda=0.5)
```

Apply the soft shrinkage function element-wise to the input Tensor

  SoftShrinkage operator:
```
       ⎧ x - lambda, if x >  lambda
f(x) = ⎨ x + lambda, if x < -lambda
       ⎩ 0, otherwise
```

**Parameters:**

**lambda**     - a factor, default is 0.5

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.SoftShrink
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val activation = SoftShrink()
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
-0.5	1.5	2.5
-1.5	2.5	3.5
-2.5	3.5	4.5
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

println(grad)
3.0	4.0	5.0
2.0	3.0	4.0
1.0	2.0	3.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```
**Scala example:**
```python
activation = SoftShrink()
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
[[-0.5  1.5  2.5]
 [-1.5  2.5  3.5]
 [-2.5  3.5  4.5]]

print grad
[[ 3.  4.  5.]
 [ 2.  3.  4.]
 [ 1.  2.  5.]]
```

