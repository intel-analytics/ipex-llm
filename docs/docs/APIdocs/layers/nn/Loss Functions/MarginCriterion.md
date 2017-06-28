## MarginCriterion ##

**Scala:**
```scala
criterion = MarginCriterion(margin=1.0, sizeAverage=true)
```
**Python:**
```python
criterion = MarginCriterion(margin=1.0, sizeAverage=true, bigdl_type="float")
```

Creates a criterion that optimizes a two-class classification hinge loss (margin-based loss) between input x (a Tensor of dimension 1) and output y.
 * @param margin if unspecified, is by default 1.
 * @param sizeAverage whether to average the loss, is by default true

**Scala example:**
```scala
val criterion = MarginCriterion[Float](margin=1.0, sizeAverage=true)

val input = Tensor[Float](3, 2).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.33753583      0.3575501
0.23477706      0.7240361
0.92835575      0.4737949
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]

val target = Tensor[Float](3, 2).rand()
target: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.27280563      0.7022703
0.3348442       0.43332106
0.08935371      0.17876455
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]

criterion.forward(input, target)
res5: Float = 0.84946966
```

**Python example:**
```python
criterion = MarginCriterion(margin=1.0,size_average=True,bigdl_type="float")
input = np.random.rand(3, 2)
array([[ 0.20824672,  0.67299837],
       [ 0.80561452,  0.19564743],
       [ 0.42501441,  0.19408184]])
       
target = np.random.rand(3, 2)
array([[ 0.67882632,  0.61257846],
       [ 0.10111138,  0.75225082],
       [ 0.60404296,  0.31373273]])
       
criterion.forward(input, target)
0.8166871
```
