## AddConstant ##

**Scala:**
```scala
val module = AddConstant(constant_scalar,inplace= false)
```
**Python:**
```python
module = AddConstant(constant_scalar,inplace=False,bigdl_type="float")
```

Element wise add a constant scalar to input tensor
* @param constant_scalar constant value
* @param inplace Can optionally do its operation in-place without using extra state memory
 
**Scala example:**
```scala
val module = AddConstant(3.0)
val input = Tensor(2,3).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.40684703      0.077655114     0.42314094
0.55392265      0.8650696       0.3621729
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

module.forward(input)
res11: com.intel.analytics.bigdl.tensor.Tensor[Float] =
3.406847        3.077655        3.423141
3.5539227       3.8650696       3.3621728
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]

```

**Python example:**
```python
module = AddConstant(3.0,inplace=False,bigdl_type="float")
input = np.array([[1, 2, 3],[4, 5, 6]])
module.forward(input)
[array([
[ 4.,  5.,  6.],
[ 7.,  8.,  9.]], dtype=float32)]
```
