## LSTM ##

**Scala:**
```scala
val lstm = LSTM(inputSize, hiddenSize)
```
**Python:**
```python
lstm = LSTM(input_size, hidden_size)
```

Long Short Term Memory architecture.

Ref:

1. http://arxiv.org/pdf/1303.5778v1 (blueprint for this module)
2. http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf
3. http://arxiv.org/pdf/1503.04069v1.pdf
4. https://github.com/wojzaremba/lstm

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val hiddenSize = 4
val inputSize = 6
val outputSize = 5
val seqLength = 5
val seed = 100

RNG.setSeed(seed)
val input = Tensor(Array(1, seqLength, inputSize))
val labels = Tensor(Array(1, seqLength))
for (i <- 1 to seqLength) {
  val rdmLabel = Math.ceil(RNG.uniform(0, 1) * outputSize).toInt
  val rdmInput = Math.ceil(RNG.uniform(0, 1) * inputSize).toInt
  input.setValue(1, i, rdmInput, 1.0f)
  labels.setValue(1, i, rdmLabel)
}

println(input)
val rec = Recurrent(hiddenSize)
val model = Sequential().add(
  rec.add(
      LSTM(inputSize, hiddenSize))).add(
        TimeDistributed(Linear(hiddenSize, outputSize)))

val criterion = TimeDistributedCriterion(
  CrossEntropyCriterion(), false)

val sgd = new SGD(learningRate=0.1, learningRateDecay=5e-7, weightDecay=0.1, momentum=0.002)

val (weight, grad) = model.getParameters()

val output = model.forward(input).toTensor
val _loss = criterion.forward(output, labels)
model.zeroGradParameters()
val gradInput = criterion.backward(output, labels)
model.backward(input, gradInput)

def feval(x: Tensor[Float]): (Float, Tensor[Float]) = {
  val output = model.forward(input).toTensor
  val _loss = criterion.forward(output, labels)
  model.zeroGradParameters()
  val gradInput = criterion.backward(output, labels)
  model.backward(input, gradInput)
  (_loss, grad)
}

var loss: Array[Float] = null
for (i <- 1 to 100) {
  loss = sgd.optimize(feval, weight)._2
  println(s"${i}-th loss = ${loss(0)}")
}
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

hidden_size = 4
input_size = 6
output_size = 5
seq_length = 5

input = np.random.uniform(0, 1, [1, seq_length, input_size]).astype("float32")
labels = np.random.uniform(1, 5, [1, seq_length]).astype("int")

print labels
print input

rec = Recurrent()
rec.add(LSTM(input_size, hidden_size))

model = Sequential()
model.add(rec)
model.add(TimeDistributed(Linear(hidden_size, output_size)))

criterion = TimeDistributedCriterion(CrossEntropyCriterion(), False)

sgd = SGD(learningrate=0.1, learningrate_decay=5e-7)

weight, grad = model.parameters()

output = model.forward(input)
loss = criterion.forward(input, labels)
gradInput = criterion.backward(output, labels)
model.backward(input, gradInput)
```
