## MultiCriterion ##

**Scala:**
```scala
Scala code, how to new an instance
```
**Python:**
```python
Python cod, how to new an instance
```

MultiCriterion is a weighted sum of other criterions each applied to the same input and target

**Scala example:**
```scala
val module = MultiCriterion()
val nll = ClassNLLCriterion()
val mse = MSECriterion()
module.add(nll, 0.5)
module.add(mse)

val input = Tensor(5).randn()
val target = Tensor(5)
target(Array(1)) = 1
target(Array(2)) = 2
target(Array(3)) = 3
target(Array(4)) = 2
target(Array(5)) = 1

> println(input)
0.9682213801388531
0.35258855644097503
0.04584479998452568
-0.21781499692588918
-1.02721844006879
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 5]

> module.forward(input, target)
res0: Double = 3.609954360965033
```

**Python example:**
```python
module = MultiCriterion()
nll = ClassNLLCriterion()
mse = MSECriterion()
module.add(nll, 0.5)
module.add(mse)

input = np.array([0.9682213801388531,
0.35258855644097503,
0.04584479998452568,
-0.21781499692588918,
-1.02721844006879])
target = np.array([1, 2, 3, 2, 1])

> module.forward(input, target)
3.6099546
```
