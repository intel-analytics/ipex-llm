## L1HingeEmbeddingCriterion ##

**Scala:**
```scala
val model = L1HingeEmbeddingCriterion(margin)
```
**Python:**
```python
model = L1HingeEmbeddingCriterion(margin)
```

Creates a criterion that measures the loss given an input ``` x = {x1, x2} ```, a table of two Tensors, and a label y (1 or -1).
This is used for measuring whether two inputs are similar or dissimilar, using the L1 distance, and is typically used for learning nonlinear embeddings or semi-supervised learning.
```
             ⎧ ||x1 - x2||_1,                  if y ==  1
loss(x, y) = ⎨
             ⎩ max(0, margin - ||x1 - x2||_1), if y == -1
```
The margin has a default value of 1, or can be set in the constructor.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val model = L1HingeEmbeddingCriterion(0.6)
val input1 = Tensor(T(1.0f, -0.1f))
val input2 = Tensor(T(2.0f, -0.2f))
val input = T(input1, input2)
val target = Tensor(1)
target(Array(1)) = 1.0f

val output = model.forward(input, target)

scala> print(output)
1.1
```

**Python example:**
```python
model = L1HingeEmbeddingCriterion(0.6)
input1 = np.array(1.0, -0.1)
input2 = np.array(2.0, -0.2)
input = [input1, input2]
target = np.array([1.0])

output = model.forward(input, target)

>>> print output
1.1
```
