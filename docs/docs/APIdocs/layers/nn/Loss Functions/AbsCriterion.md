## AbsCriterion ##

**Scala:**
```scala
val criterion = AbsCriterion(sizeAverage)
```
**Python:**
```python
criterion = AbsCriterion(sizeAverage)
```

Measures the mean absolute value of the element-wise difference between input and target

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val criterion = AbsCriterion()
val input = Tensor(T(1.0f, 2.0f, 3.0f))
val target = Tensor(T(4.0f, 5.0f, 6.0f))
val output = criterion.forward(input, target)

scala> print(output)
3.0
```

**Python example:**
```python
criterion = AbsCriterion()
input = np.array([1.0, 2.0, 3.0])
target = np.array([4.0, 5.0, 6.0])
output=criterion.forward(input, target)

>>> print output
3.0
```
