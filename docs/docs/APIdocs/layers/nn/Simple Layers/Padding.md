## Padding ##

**Scala:**
```scala
val module = Padding(dim, pad, nInputDim)
```
**Python:**
```python
module = Padding(dim, pad, n_input_dim)
```

This module adds pad units of padding to dimension dim of the input. If pad is negative,
padding is added to the left, otherwise, it is added to the right of the dimension.
The input to this layer is expected to be a tensor, or a batch of tensors;
when using mini-batch, a batch of sample tensors will be passed to the layer and
the user need to specify the number of dimensions of each sample tensor in the
batch using nInputDims.

 * @param dim the dimension to be applied padding operation
 * @param pad num of the pad units
 * @param nInputDim specify the number of dimensions that this module will receive
                  If it is more than the dimension of input tensors, the first dimension
                  would be considered as batch size
 * @param value padding value, default is 0

**Scala example:**
```scala
val module = Padding[Double](1, -1, 3, 0.9)
val input = Tensor[Double](3, 2, 1).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,.,.) =
0.5128940341528505
0.3143353720661253

(2,.,.) =
0.8289588657207787
0.9162355500739068

(3,.,.) =
0.04309450043365359
0.6807688088156283

[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x2x1]

module.forward(input)
res7: com.intel.analytics.bigdl.tensor.Tensor[Double] =
(1,.,.) =
0.9
0.9

(2,.,.) =
0.5128940341528505
0.3143353720661253

(3,.,.) =
0.8289588657207787
0.9162355500739068

(4,.,.) =
0.04309450043365359
0.6807688088156283

[com.intel.analytics.bigdl.tensor.DenseTensor of size 4x2x1]
```

**Python example:**
```python
module = Padding(1, -1, 3, 0.9)
input = np.random.rand(3, 2, 1)
module.forward(input)
[array([
[[ 0.89999998],
 [ 0.89999998]],

[[ 0.97105861],
 [ 0.09420794]],

[[ 0.28587413],
 [ 0.52355504]],

[[ 0.47290906],
 [ 0.77669364]]], dtype=float32)]
```
