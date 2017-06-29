## Normalize ##

**Scala:**
```scala
val module = Normalize(p,eps=1e-10)
```
**Python:**
```python
module = Normalize(p,eps=1e-10,bigdl_type="float")
```

Normalizes the input Tensor to have unit L_p norm. The smoothing parameter eps prevents
division by zero when the input contains all zero elements (default = 1e-10).
The input can be 1d, 2d or 4d. If the input is 4d, it should follow the format (n, c, h, w) where n is the batch number,
c is the channel number, h is the height and w is the width
 * @param p L_p norm
 * @param eps smoothing parameter

**Scala example:**
```scala
val module = Normalize(2.0,eps=1e-10)
val input = Tensor(2,3).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.7075603       0.084298864     0.91339105
0.22373432      0.8704987       0.6936567
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

module.forward(input)
res8: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.6107763       0.072768        0.7884524
0.19706465      0.76673317      0.61097115
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
module = Normalize(2.0,eps=1e-10,bigdl_type="float")
input = np.array([[1, 2, 3],[4, 5, 6]])
module.forward(input)
[array([
[ 0.26726124,  0.53452247,  0.80178368],
[ 0.45584232,  0.56980288,  0.68376344]], dtype=float32)]
```
