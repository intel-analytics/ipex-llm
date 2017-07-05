## L1Cost ##

**Scala:**
```scala
val layer = L1Cost[Float]()
```
**Python:**
```python
layer = L1Cost()
```

Compute L1 norm for input, and sign of input

**Scala example:**
```scala
val layer = L1Cost[Float]()
val input = Tensor[Float](2, 2).rand
val target = Tensor[Float](2, 2).rand

val output = layer.forward(input, target)
val gradInput = layer.backward(input, target)

> println(input)
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.48145306      0.476887
0.23729686      0.5169516
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

> println(target)
target: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.42999148      0.22272833
0.49723643      0.17884709
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

> println(output)
output: Float = 1.7125885
> println(gradInput)
gradInput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0     1.0
1.0     1.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
layer = L1Cost()

input = np.random.uniform(0, 1, (2, 2)).astype("float32")
target = np.random.uniform(0, 1, (2, 2)).astype("float32")

output = layer.forward(input, target)
gradInput = layer.backward(input, target)

> output
2.522411
> gradInput
[array([[ 1.,  1.],
        [ 1.,  1.]], dtype=float32)]
```
