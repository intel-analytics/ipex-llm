## Normalize ##

**Scala:**
```scala
val module = Normalize(p)
```
**Python:**
```python
module = Normalize(p)
```

Normalizes the input Tensor to have unit L_p norm. The smoothing parameter eps prevents
division by zero when the input contains all zero elements (default = 1e-10).
The input can be 1d, 2d or 4d. If the input is 4d, it should follow the format (n, c, h, w) where n is the batch number,
c is the channel number, h is the height and w is the width
 * @param p L_p norm
 * @param eps smoothing parameter

**Scala example:**
```scala
val module = Normalize[Double](2.0)
val input = Tensor[Double](2,3).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.8007086841389537      0.19004820031113923     0.8484807822387666
0.5260537276044488      0.44199418602511287     0.7931551630608737
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2x3]

module.forward(input)
res4: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.6774067923765057      0.16078249720513227     0.7178224196575933
0.5013022611934496      0.4211978230774036      0.755836249902305
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
module = Normalize(2.0)
input = np.array([[1, 2, 3],[4, 5, 6]])
module.forward(input)
[array([
[ 0.26726124,  0.53452247,  0.80178368],
[ 0.45584232,  0.56980288,  0.68376344]], dtype=float32)]
```
