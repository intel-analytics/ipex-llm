## MarginCriterion ##

**Scala:**
```scala
criterion = MarginCriterion()
```
**Python:**
```python
criterion = MarginCriterion()
```

Creates a criterion that optimizes a two-class classification hinge loss (margin-based loss) between input x (a Tensor of dimension 1) and output y.
 * @param margin if unspecified, is by default 1.
 * @param sizeAverage whether to average the loss, is by default true

**Scala example:**
```scala
val criterion = MarginCriterion[Double]()
val input = Tensor[Double](3, 2).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.544831171631813       0.677923800656572
0.10097078117541969     0.7837557627353817
0.9371688910759985      0.065070073120296
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x2]

val target = Tensor[Double](3, 2).rand()
target: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.07769605284556746     0.2338713244535029
0.5209604250267148      0.03824349492788315
0.7666055841837078      0.9896160187199712
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x2]

criterion.forward(input, target)
res0: Double = 0.8222855331373905
```

**Python example:**
```python
criterion = MarginCriterion()
input = np.random.rand(3, 2)
target = np.random.rand(3, 2)
criterion.forward(input, target)
0.8955021
```
