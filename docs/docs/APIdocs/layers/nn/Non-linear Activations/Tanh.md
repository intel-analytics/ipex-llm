## Tanh
**Scala:**
```scala
val activation = Tanh()
```
**Python:**
```python
activation = Tanh()
```

Applies the Tanh function element-wise to the input Tensor,
thus outputting a Tensor of the same dimension.
Tanh is defined as
```
f(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x)).
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Tanh
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val activation = Tanh()
val input = Tensor(T(
  T(1f, 2f, 3f),
  T(2f, 3f, 4f),
  T(3f, 4f, 5f)
))

val gradOutput = Tensor(T(
  T(3f, 4f, 5f),
  T(2f, 3f, 4f),
  T(1f, 2f, 3f)
))

val output = activation.forward(input)
val grad = activation.backward(input, gradOutput)

println(output)
0.7615942	0.9640276	0.9950548
0.9640276	0.9950548	0.9993293
0.9950548	0.9993293	0.9999092
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

println(grad)
1.259923	0.28260326	0.049329996
0.14130163	0.029597998	0.0053634644
0.009865999	0.0026817322	5.4466724E-4
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**
```python
activation = Tanh()
input = np.array([
  [1.0, 2.0, 3.0],
  [2.0, 3.0, 4.0],
  [3.0, 4.0, 5.0]
])

gradOutput = np.array([
  [3.0, 4.0, 5.0],
  [2.0, 3.0, 4.0],
  [1.0, 2.0, 5.0]
])

output = activation.forward(input)
grad = activation.backward(input, gradOutput)

print output
[[ 0.76159418  0.96402758  0.99505478]
 [ 0.96402758  0.99505478  0.99932933]
 [ 0.99505478  0.99932933  0.99990922]]

print grad
[[  1.25992298e+00   2.82603264e-01   4.93299961e-02]
 [  1.41301632e-01   2.95979977e-02   5.36346436e-03]
 [  9.86599922e-03   2.68173218e-03   9.07778740e-04]]
```