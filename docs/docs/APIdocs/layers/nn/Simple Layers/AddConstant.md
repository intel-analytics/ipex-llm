## AddConstant ##

**Scala:**
```scala
val module = AddConstant(constant_scalar)
```
**Python:**
```python
module = AddConstant(constant_scalar)
```

Element wise add a constant scalar to input tensor
* @param constant_scalar constant value
* @param inplace Can optionally do its operation in-place without using extra state memory
 
**Scala example:**
```scala
val module = AddConstant[Double](3.0)
val input = Tensor[Double](2,3).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.7599249668419361      0.8327403955627233      0.7157071044202894
0.7422519933898002      0.09718431462533772     0.3686083541251719
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2x3]

module.forward(input)
res5: com.intel.analytics.bigdl.tensor.Tensor[Double] =
3.759924966841936       3.8327403955627233      3.7157071044202894
3.7422519933898 3.0971843146253377      3.368608354125172
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
module = AddConstant(3.0)
input = np.array([[1, 2, 3],[4, 5, 6]])
module.forward(input)
[array([
[ 4.,  5.,  6.],
[ 7.,  8.,  9.]], dtype=float32)]
```
