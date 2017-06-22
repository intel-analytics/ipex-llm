## AbsCriterion ##

**Scala:**
```scala
val criterion = AbsCriterion[Float]()
```
**Python:**
```python
criterion = AbsCriterion()
```

measures the mean absolute value of the element-wise difference between input

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
val criterion = AbsCriterion[Float]()

val input = Tensor[Float](3).rand()
val target = Tensor[Float](3).rand()
val output = criterion.forward(input, target)
```
output is
```
output: Float = 0.33056465
```

**Python example:**
```python
criterion = AbsCriterion()
input = np.array([0.9682213801388531,
0.35258855644097503,
0.04584479998452568,
-0.21781499692588918,
-1.02721844006879])
target = np.array([1, 2, 3, 2, 1])
output=criterion.forward(input, target)
```
output is
```
1.7756758
```
