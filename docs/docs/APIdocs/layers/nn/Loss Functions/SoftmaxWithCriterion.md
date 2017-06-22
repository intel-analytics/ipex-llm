## SoftmaxWithCriterion ##

**Scala:**
```scala
val model = SoftmaxWithCriterion[T](ignoreLabel, normalizeMode)
```
**Python:**
```python
model = SoftmaxWithCriterion(ignoreLabel, normalizeMode)
```

Computes the multinomial logistic loss for a one-of-many classification task,
passing real-valued predictions through a softmax to get a probability distribution over classes.
It should be preferred over separate SoftmaxLayer + MultinomialLogisticLossLayer
as its gradient computation is more numerically stable.
- param ignoreLabel   (optional) Specify a label value that should be ignored when computing the loss.
- param normalizeMode How to normalize the output loss.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

val input = Tensor[Float](1, 5, 2, 3).rand()
val target = Tensor(Storage(Array(2.0f, 4.0f, 2.0f, 4.0f, 1.0f, 2.0f))).resize(1, 1, 2, 3)

val model = SoftmaxWithCriterion[Float]()
val output = model.forward(input, target)
```
output is
```
output: Float = 10.494613
```
**Python example:**
```python
input = np.random.randn(1, 5, 2, 3)
target = np.array([[[[2.0, 4.0, 2.0], [4.0, 1.0, 2.0]]]])

model = SoftmaxWithCriterion()
output = model.forward(input, target)
```
output is
```
2.1241186
```
