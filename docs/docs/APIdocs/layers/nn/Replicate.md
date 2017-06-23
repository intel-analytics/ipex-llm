## Replicate ##

**Scala:**
```scala
val module = Replicate(
  nFeatures : Int,
  dim : Int = 1,
  nDim : Int = Int.MaxValue)
```
**Python:**
```python
module = Replicate(
  n_features,
  dim=1,
  n_dim=INTMAX)
```
Replicate repeats input `nFeatures` times along its `dim` dimension

Notice: No memory copy, it set the stride along the `dim`-th dimension to zero.

**Scala example:**
```scala
val module = Replicate(4, 1, 2)

println(module.forward(Tensor.range(1, 6, 1).resize(1, 2, 3)))
```
Output is
```
com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
1.0	2.0	3.0
4.0	5.0	6.0

(1,2,.,.) =
1.0	2.0	3.0
4.0	5.0	6.0

(1,3,.,.) =
1.0	2.0	3.0
4.0	5.0	6.0

(1,4,.,.) =
1.0	2.0	3.0
4.0	5.0	6.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x4x2x3]
```

**Python example:**
```python
module = Replicate(4, 1, 2)

print(module.forward(np.arange(1, 7, 1).reshape(1, 2, 3)))
```
Output is 
```
[array([[[[ 1.,  2.,  3.],
         [ 4.,  5.,  6.]],

        [[ 1.,  2.,  3.],
         [ 4.,  5.,  6.]],

        [[ 1.,  2.,  3.],
         [ 4.,  5.,  6.]],

        [[ 1.,  2.,  3.],
         [ 4.,  5.,  6.]]]], dtype=float32)]
```
