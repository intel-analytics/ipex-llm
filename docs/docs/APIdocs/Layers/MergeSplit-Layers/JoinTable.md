## JoinTable
**Scala:**
```scala
val layer = JoinTable(dimension, nInputDims)
```
**Python:**
```python
layer = JoinTable(dimension, n_input_dims)
```

It is a table module which takes a table of Tensors as input and
outputs a Tensor by joining them together along the dimension `dimension`.


The input to this layer is expected to be a tensor, or a batch of tensors;
when using mini-batch, a batch of sample tensors will be passed to the layer and
the user need to specify the number of dimensions of each sample tensor in the
batch using `nInputDims`.

**Parameters:**

**dimension**  - to be join in this dimension

**nInputDims** - specify the number of dimensions that this module will receiveIf it is more than the dimension of input tensors, the first dimensionwould be considered as batch size

```
+----------+             +-----------+
| {input1, +-------------> output[1] |
|          |           +-----------+-+
|  input2, +-----------> output[2] |
|          |         +-----------+-+
|  input3} +---------> output[3] |
+----------+         +-----------+
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.JoinTable
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val layer = JoinTable(2, 2)
val input1 = Tensor(T(
  T(
    T(1f, 2f, 3f),
    T(2f, 3f, 4f),
    T(3f, 4f, 5f))
))

val input2 = Tensor(T(
  T(
    T(3f, 4f, 5f),
    T(2f, 3f, 4f),
    T(1f, 2f, 3f))
))

val input = T(input1, input2)

val gradOutput = Tensor(T(
  T(
    T(1f, 2f, 3f, 3f, 4f, 5f),
    T(2f, 3f, 4f, 2f, 3f, 4f),
    T(3f, 4f, 5f, 1f, 2f, 3f)
)))

val output = layer.forward(input)
val grad = layer.backward(input, gradOutput)

println(output)
(1,.,.) =
1.0	2.0	3.0	3.0	4.0	5.0
2.0	3.0	4.0	2.0	3.0	4.0
3.0	4.0	5.0	1.0	2.0	3.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x6]

println(grad)
 {
	2: (1,.,.) =
	   3.0	4.0	5.0
	   2.0	3.0	4.0
	   1.0	2.0	3.0

	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x3]
	1: (1,.,.) =
	   1.0	2.0	3.0
	   2.0	3.0	4.0
	   3.0	4.0	5.0

	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x3]
 }
```

**Python example:**
```python
layer = JoinTable(2, 2)
input1 = np.array([
 [
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0]
  ]
])

input2 = np.array([
  [
    [3.0, 4.0, 5.0],
    [2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0]
  ]
])

input = [input1, input2]

gradOutput = np.array([
  [
    [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
    [2.0, 3.0, 4.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 1.0, 2.0, 3.0]
  ]
])

output = layer.forward(input)
grad = layer.backward(input, gradOutput)

print output
[[[ 1.  2.  3.  3.  4.  5.]
  [ 2.  3.  4.  2.  3.  4.]
  [ 3.  4.  5.  1.  2.  3.]]]

print grad
[array([[[ 1.,  2.,  3.],
        [ 2.,  3.,  4.],
        [ 3.,  4.,  5.]]], dtype=float32), array([[[ 3.,  4.,  5.],
        [ 2.,  3.,  4.],
        [ 1.,  2.,  3.]]], dtype=float32)]
```
