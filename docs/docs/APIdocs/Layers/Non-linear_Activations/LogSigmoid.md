## LogSigmoid
**Scala:**
```scala
val activation = LogSigmoid()
```
**Python:**
```python
activation = LogSigmoid()
```

This class is a activation layer corresponding to the non-linear function sigmoid function:
```
f(x) = Log(1 / (1 + e ^ (-x)))
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.LogSigmoid
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val activation = LogSigmoid()
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
-0.3132617	-0.12692802	-0.04858735
-0.12692802	-0.04858735	-0.01814993
-0.04858735	-0.01814993	-0.0067153485
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

println(grad)
0.8068244	0.47681168	0.23712938
0.23840584	0.14227761	0.07194484
0.047425874	0.03597242	0.020078553
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**
```python
activation = LogSigmoid()
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
[[-0.31326169 -0.12692802 -0.04858735]
 [-0.12692802 -0.04858735 -0.01814993]
 [-0.04858735 -0.01814993 -0.00671535]]

print grad
[[ 0.80682439  0.47681168  0.23712938]
 [ 0.23840584  0.14227761  0.07194484]
 [ 0.04742587  0.03597242  0.03346425]]
```

