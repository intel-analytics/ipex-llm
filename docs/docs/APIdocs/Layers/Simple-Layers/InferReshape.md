## InferReshape

**Scala:**
```scala
val layer = InferReshape(size, batchMode = false)
```
**Python:**
```python
layer = InferReshape(size, batch_mode=False)
```

Reshape the input tensor with automatic size inference support.
Positive numbers in the `size` argument are used to reshape the input to the
corresponding dimension size.

There are also two special values allowed in `size`:

   1. `0` means keep the corresponding dimension size of the input unchanged.
      i.e., if the 1st dimension size of the input is 2,
      the 1st dimension size of output will be set as 2 as well.
   2. `-1` means infer this dimension size from other dimensions.
      This dimension size is calculated by keeping the amount of output elements
      consistent with the input.
      Only one `-1` is allowable in `size`.

For example,
```
   Input tensor with size: (4, 5, 6, 7)
   -> InferReshape(Array(4, 0, 3, -1))
   Output tensor with size: (4, 5, 3, 14)
```

The 1st and 3rd dim are set to given sizes, keep the 2nd dim unchanged,
and inferred the last dim as 14.

**Parameters:**

**size**      -the target tensor size

**batchMode** -whether in batch mode

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.InferReshape
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat

val layer = InferReshape(Array(0, 3, -1))
val input = Tensor(1, 2, 3).rand()
val gradOutput = Tensor(1, 3, 2).rand()

val output = layer.forward(input)
val grad = layer.backward(input, gradOutput)

println(output)
(1,.,.) =
0.8170822	0.40073588
0.49389255	0.3782435
0.42660004	0.5917206

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x2]

println(grad)
(1,.,.) =
0.8294597	0.57101834	0.90910035
0.32783163	0.30494633	0.7339092

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x3]
```

**Python example:**
```python
layer = InferReshape([0, 3, -1])
input = np.random.rand(1, 2, 3)

gradOutput = np.random.rand(1, 3, 2)

output = layer.forward(input)
grad = layer.backward(input, gradOutput)

print output
[[[ 0.68635464  0.21277553]
  [ 0.13390459  0.65662414]
  [ 0.1021723   0.92319047]]]

print grad
[[[ 0.84927064  0.55205333  0.25077972]
  [ 0.76105869  0.30828172  0.1237276 ]]]
```