## RNN ##

**Scala:**
```scala
val rnnCell = RnnCell[Double](inputSize, hiddenSize, activation, wRegularizer, uRegularizer, bRegularizer)
```
**Python:**
```python
rnnCell = RnnCell(input_size, hidden_size, Tanh(), w_regularizer, u_regularizer, b_regularizer)
```

Implementation of vanilla recurrent neural network cell
i2h: weight matrix of input to hidden units
h2h: weight matrix of hidden units to themselves through time
The updating is defined as:
h_t = f(i2h * x_t + h2h * h_{t-1})

**Parameters:**
* **inputSize** - input size. Default: 4
* **hiddenSize** - hidden layer size. Default: 3
* **activation** - activation function f for non-linearity
* **wRegularizer** - instance of `Regularizer`(eg. L1 or L2 regularization), applied to the input weights matrices. Default: null
* **uRegularizer** - instance of `Regularizer`(eg. L1 or L2 regularization), applied to the recurrent weights matrices. Default: null
* **bRegularizer** - instance of `Regularizer`(eg. L1 or L2 regularization), applied to the bias. Default: null

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val hiddenSize = 2
val inputSize = 2
val outputSize = 2
val seqLength = 2
val input = Tensor(T(
  T(1.0f, 2.0f),
  T(2.0f, 3.0f)
)).resize(Array(1, seqLength, inputSize))
val target = Tensor(T(
  T(2.0f, 3.0f),
  T(4.0f, 5.0f)
)).resize(Array(1, seqLength, inputSize))
val rec = Recurrent()

val model = Sequential()
    .add(rec.add(RnnCell(inputSize, hiddenSize, Tanh())))
    .add(TimeDistributed(Linear(hiddenSize, outputSize)))
val output = model.forward(input)
val gradient = model.backward(input, target)
-> print(output)
# There's random factor. An output could be
(1,.,.) =
0.41442442      0.1663357       
0.5339842       0.57332826      

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x2]
-> print(gradient)
# There's random factor. An output could be
(1,.,.) =
1.1512008       2.181274        
-0.4805725      1.6620052       

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
hidden_size = 2
input_size = 2
output_size = 2
seq_length = 2
input = np.array([[
  [1.0, 2.0],
  [2.0, 3.0]
]])
target = np.array([[
  [2.0, 3.0],
  [4.0, 5.0]
]])
rec = Recurrent()

model = Sequential() \
    .add(rec.add(RnnCell(input_size, hidden_size, Tanh()))) \
    .add(TimeDistributed(Linear(hidden_size, output_size)))
output = model.forward(input)
gradient = model.backward(input, target)
-> print output
# There's random factor. An output could be
[[[-0.67860311  0.80307233]
  [-0.77462083  0.97191858]]]

-> print gradient
# There's random factor. An output could be
[[[-0.90771425  1.24791598]
  [-0.70141178  0.97821164]]]
```
