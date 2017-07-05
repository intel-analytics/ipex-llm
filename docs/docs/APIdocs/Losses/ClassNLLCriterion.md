## ClassNLLCriterion

**Scala:**
```scala
val criterion = ClassNLLCriterion(weights = null, sizeAverage = true)
```
**Python:**
```python
criterion = ClassNLLCriterion(weights=None, size_average=True)
```

The negative log likelihood criterion. It is useful to train a classification problem with n
classes. If provided, the optional argument weights should be a 1D Tensor assigning weight to
each of the classes. This is particularly useful when you have an unbalanced training set.

The input given through a `forward()` is expected to contain log-probabilities of each class:
input has to be a 1D Tensor of size `n`. Obtaining log-probabilities in a neural network is easily
achieved by adding a `LogSoftMax` layer in the last layer of your neural network. You may use
`CrossEntropyCriterion` instead, if you prefer not to add an extra layer to your network. This
criterion expects a class index (1 to the number of class) as target when calling
`forward(input, target)` and `backward(input, target)`.

 The loss can be described as:
     loss(x, class) = -x[class]
 or in the case of the weights argument it is specified as follows:
     loss(x, class) = -weights[class] * x[class]
 Due to the behaviour of the backend code, it is necessary to set sizeAverage to false when
 calculating losses in non-batch mode.

 By default, the losses are averaged over observations for each minibatch. However, if the field
 `sizeAverage` is set to false, the losses are instead summed for each minibatch.

**Parameters:**

**weights**     - weights of each element of the input

**sizeAverage** - A boolean indicating whether normalizing by the number of elements in the input.
                  Default: true

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val criterion = ClassNLLCriterion()
val input = Tensor(T(
              T(1f, 2f, 3f),
              T(2f, 3f, 4f),
              T(3f, 4f, 5f)
          ))

val target = Tensor(T(1f, 2f, 3f))

val loss = criterion.forward(input, target)
val grad = criterion.backward(input, target)

print(loss)
-3.0
println(grad)
-0.33333334	0.0	0.0
0.0	-0.33333334	0.0
0.0	0.0	-0.33333334
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *

criterion = ClassNLLCriterion()
input = np.array([
              [1.0, 2.0, 3.0],
              [2.0, 3.0, 4.0],
              [3.0, 4.0, 5.0]
          ])

target = np.array([1.0, 2.0, 3.0])

loss = criterion.forward(input, target)
gradient= criterion.backward(input, target)

print loss
-3.0
print gradient
-3.0
[[-0.33333334  0.          0.        ]
 [ 0.         -0.33333334  0.        ]
 [ 0.          0.         -0.33333334]]
```

