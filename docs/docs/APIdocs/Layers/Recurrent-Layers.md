## Recurrent ##

**Scala:**
```scala
val module = Recurrent()
```
**Python:**
```python
module = Recurrent()
```

Recurrent module is a container of rnn cells. Different types of rnn cells can be added using add() function.  

Recurrent supports returning final state and cell status of its rnn cells by using getFinalStateAndCellStatus. output of getFinalStateAndCellStatus
is a tuple. The first element is the output state at last time while the second elements is the cell status.  

If contained cell is simple rnn, finalstate is `batch x hiddenSize`. cell is None  
If contained cell is lstm, finalstate is `batch x hiddenSize`. cell is `batch x hiddenSize`  
If contained cell is convlstm2D, finalstate is `batch x outputPlane x height x width`. cell is `batch x outputPlane x height x width`  
If contained cell is convlstm3D, finalstate is `batch x outputPlane x height x width x length`. cell is `batch x outputPlane x height x width x length`

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val hiddenSize = 4
val inputSize = 5
val module = Recurrent().add(RnnCell(inputSize, hiddenSize, Tanh()))
val input = Tensor(Array(1, 5, inputSize))
for (i <- 1 to 5) {
  val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0)*inputSize).toInt
  input.setValue(1, i, rdmInput, 1.0f)
}

val output = module.forward(input)

val (finalState, cellStatus) = module.getFinalStateAndCellStatus()

> input
(1,.,.) =
0.0	0.0	0.0	1.0	0.0	
0.0	0.0	0.0	0.0	1.0	
0.0	1.0	0.0	0.0	0.0	
0.0	1.0	0.0	0.0	0.0	
0.0	0.0	0.0	0.0	1.0	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x5x5]

> output
(1,.,.) =
0.23312	-0.5702369	-0.29894134	-0.46780553	
-0.020703634	-0.6821252	-0.71641463	-0.3367952	
0.031236319	-0.29233444	-0.730908	0.13494356	
-0.22310422	-0.25562853	-0.59091455	-0.25055867	
0.007001166	-0.7096118	-0.778529	-0.47429603	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5x4]

> finalState
0.007001166	-0.7096118	-0.778529	-0.47429603

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x4]

> cellStatus
null

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

hiddenSize = 4
inputSize = 5
module = Recurrent().add(RnnCell(inputSize, hiddenSize, Tanh()))
input = np.zeros((1, 5, 5))
input[0][0][4] = 1
input[0][1][0] = 1
input[0][2][4] = 1
input[0][3][3] = 1
input[0][4][0] = 1

output = module.forward(input)

res = module.get_finalState_cellStatus()

> input
[[[ 0.  0.  0.  0.  1.]
  [ 1.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  1.]
  [ 0.  0.  0.  1.  0.]
  [ 1.  0.  0.  0.  0.]]]

> output
[[[-0.43169451 -0.27838707  0.41472727  0.4450382 ]
  [-0.10717546  0.59218317  0.67959404  0.62824875]
  [-0.56745911 -0.31170678  0.44158491  0.31494498]
  [ 0.13328044  0.41262615  0.37388939  0.10983802]
  [-0.51452565  0.13222042  0.59192103  0.8393243 ]]]

> res[0]
# final state
[[-0.51452565  0.13222042  0.59192103  0.8393243 ]]

> res[1]
# cell status
None
```
---
## BiRecurrent ##

**Scala:**
```scala
val module = BiRecurrent(merge=null)
```
**Python:**
```python
module = BiRecurrent(merge=None,bigdl_type="float")
```

This layer implement a bidirectional recurrent neural network

 * `merge` concat or add the output tensor of the two RNNs. Default is add

**Scala example:**
```scala
val module = BiRecurrent(CAddTable())
.add(RnnCell(6, 4, Sigmoid()))
val input = Tensor(Array(1, 2, 6)).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.55511624      0.44330198      0.9025551       0.26096714      0.3434667       0.20060952
0.24903035      0.24026379      0.89252585      0.23025699      0.8131796       0.4013688

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x2x6]

module.forward(input)
res10: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.3577285       0.8861933       0.52908427      0.86278
1.2850789       0.82549953      0.5560188       0.81468254

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x4]
```

**Python example:**
```python
module = BiRecurrent(CAddTable()).add(RnnCell(6, 4, Sigmoid()))
input = np.random.rand(1, 2, 6)
array([[[ 0.75637438,  0.2642816 ,  0.61973312,  0.68565282,  0.73571443,
          0.17167681],
        [ 0.16439321,  0.06853251,  0.42257202,  0.42814042,  0.15706152,
          0.57866659]]])

module.forward(input)
array([[[ 0.69091094,  0.97150528,  0.9562254 ,  1.14894259],
        [ 0.83814102,  1.11358368,  0.96752423,  1.00913286]]], dtype=float32)
```
---
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

* `i2h` weight matrix of input to hidden units
* `h2h` weight matrix of hidden units to themselves through time

The updating is defined as:

```
h_t = f(i2h * x_t + h2h * h_{t-1})
```

Parameters:

* `inputSize` input size. Default: 4
* `hiddenSize`  hidden layer size. Default: 3
* `activation` activation function f for non-linearity
* `wRegularizer` instance of `Regularizer`(eg. L1 or L2 regularization), applied to the input weights matrices. Default: null
* `uRegularizer` instance of `Regularizer`(eg. L1 or L2 regularization), applied to the recurrent weights matrices. Default: null
* `bRegularizer` instance of `Regularizer`(eg. L1 or L2 regularization), applied to the bias. Default: null

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
val gradOutput = Tensor(T(
  T(2.0f, 3.0f),
  T(4.0f, 5.0f)
)).resize(Array(1, seqLength, inputSize))
val rec = Recurrent()

val model = Sequential()
    .add(rec.add(RnnCell(inputSize, hiddenSize, Tanh())))
    .add(TimeDistributed(Linear(hiddenSize, outputSize)))
val output = model.forward(input)
val gradient = model.backward(input, gradOutput)
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
grad_output = np.array([[
  [2.0, 3.0],
  [4.0, 5.0]
]])
rec = Recurrent()

model = Sequential() \
    .add(rec.add(RnnCell(input_size, hidden_size, Tanh()))) \
    .add(TimeDistributed(Linear(hidden_size, output_size)))
output = model.forward(input)
gradient = model.backward(input, grad_output)
-> print output
# There's random factor. An output could be
[[[-0.67860311  0.80307233]
  [-0.77462083  0.97191858]]]

-> print gradient
# There's random factor. An output could be
[[[-0.90771425  1.24791598]
  [-0.70141178  0.97821164]]]
```
---
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

---
## LSTMPeephole ##

**Scala:**
```scala
val model = LSTMPeephole(
  inputSize = 4,
  hiddenSize = 3,
  p = 0.0,
  wRegularizer = null,
  uRegularizer = null,
  bRegularizer = null)
```
**Python:**
```python
model = LSTMPeephole(
  input_size,
  hidden_size,
  p=0.0,
  wRegularizer=None,
  uRegularizer=None,
  bRegularizer=None)
```

Long Short Term Memory architecture with peephole.
Ref.

1. http://arxiv.org/pdf/1303.5778v1 (blueprint for this module)
2. http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf
3. http://arxiv.org/pdf/1503.04069v1.pdf
4. https://github.com/wojzaremba/lstm

Parameters:
* `inputSize` the size of each input vector
* `hiddenSize` Hidden unit size in the LSTM
* `p` is used for [[Dropout]] probability. For more details about
           RNN dropouts, please refer to
           [RnnDrop: A Novel Dropout for RNNs in ASR]
           (http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)
           [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks]
           (https://arxiv.org/pdf/1512.05287.pdf)
* `wRegularizer` instance of [[Regularizer]]
                   (eg. L1 or L2 regularization), applied to the input weights matrices.
* `uRegularizer` instance [[Regularizer]]
          (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
* `bRegularizer` instance of [[Regularizer]] applied to the bias.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

val hiddenSize = 4
val inputSize = 6
val outputSize = 5
val seqLength = 5
val batchSize = 1
               
val input = Tensor(Array(batchSize, seqLength, inputSize))
for (b <- 1 to batchSize) {
  for (i <- 1 to seqLength) {
    val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0) * inputSize).toInt
    input.setValue(b, i, rdmInput, 1.0f)
  }
}

val rec = Recurrent(hiddenSize)
val model = Sequential().add(rec.add(LSTMPeephole(inputSize, hiddenSize))).add(TimeDistributed(Linear(hiddenSize, outputSize)))
val output = model.forward(input).toTensor

scala> print(input)
(1,.,.) =
1.0	0.0	0.0	0.0	0.0	0.0	
0.0	0.0	0.0	0.0	0.0	1.0	
0.0	1.0	0.0	0.0	0.0	0.0	
0.0	0.0	0.0	0.0	0.0	1.0	
1.0	0.0	0.0	0.0	0.0	0.0	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x5x6]

scala> print(output)
(1,.,.) =
0.34764957	-0.31453514	-0.45646006	-0.42966008	-0.13651063	
0.3624894	-0.2926056	-0.4347164	-0.40951455	-0.1775867	
0.33391106	-0.29304913	-0.4748538	-0.45285955	-0.14919288	
0.35499972	-0.29385415	-0.4419502	-0.42135617	-0.17544147	
0.32911295	-0.30237123	-0.47175884	-0.4409852	-0.15733294	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5x5]
```
**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

hiddenSize = 4
inputSize = 6
outputSize = 5
seqLength = 5
batchSize = 1
               
input = np.random.randn(batchSize, seqLength, inputSize)
rec = Recurrent(hiddenSize)
model = Sequential().add(rec.add(LSTMPeephole(inputSize, hiddenSize))).add(TimeDistributed(Linear(hiddenSize, outputSize)))
output = model.forward(input)

>>> print(input)
[[[ 0.73624017 -0.91135209 -0.30627796 -1.07902111 -1.13549159  0.52868762]
  [-0.07251559 -0.45596589  1.64020513  0.53218623  1.37993166 -0.47724947]
  [-1.24958366 -1.22220259 -0.52454306  0.17382396  1.77666173 -1.2961758 ]
  [ 0.45407533  0.82944329  0.02155243  1.82168093 -0.06022129  2.23823013]
  [ 1.09100802  0.28555387 -0.94312648  0.55774033 -0.54895792  0.79885853]]]
  
>>> print(output)
[[[ 0.4034881  -0.26156989  0.46799076  0.06283229  0.11794794]
  [ 0.37359846 -0.17925361  0.31623816  0.06038529  0.10813089]
  [ 0.34150451 -0.16565879  0.25264332  0.1187657   0.05118144]
  [ 0.40773875 -0.2028828   0.24765283  0.0986848   0.12132661]
  [ 0.40263647 -0.22403356  0.38489845  0.04720671  0.1686969 ]]]
```
---
## GRU ##

**Scala:**
```scala
val gru = GRU(inputSize, outputSize, p, wRegularizer, uRegularizer, bRegularizer)
```
**Python:**
```python
gru = GRU(inputSize, outputSize, p, w_regularizer, u_regularizer, b_regularizer)
```

Gated Recurrent Units architecture. The first input in sequence uses zero value for cell and hidden state.

Ref.

1. http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
2. https://github.com/Element-Research/rnn/blob/master/GRU.lua
 
 
Parameters:

* `inputSize` the size of each input vector
* `outputSize` hidden unit size in GRU
* `p` is used for [[Dropout]] probability. For more details about
          RNN dropouts, please refer to
           [RnnDrop: A Novel Dropout for RNNs in ASR](http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)
            and [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf). Default: 0.0
* `wRegularizer` instance of `Regularizer`(eg. L1 or L2 regularization), applied to the input weights matrices. Default: null
* `uRegularizer` instance of `Regularizer`(eg. L1 or L2 regularization), applied to the recurrent weights matrices. Default: null
* `bRegularizer` instance of `Regularizer`(eg. L1 or L2 regularization), applied to the bias. Default: null

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
val gradOutput = Tensor(T(
  T(2.0f, 3.0f),
  T(4.0f, 5.0f)
)).resize(Array(1, seqLength, inputSize))
val rec = Recurrent()

val model = Sequential()
    .add(rec.add(GRU(inputSize, hiddenSize)))
    .add(TimeDistributed(Linear(hiddenSize, outputSize)))
val output = model.forward(input)
val gradient = model.backward(input, gradOutput)

-> print(output)
# There's random factor. An output could be
(1,.,.) =
0.3833429       0.0082434565    
-0.041063666    -0.08152798     

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x2]


-> print(gradient)
# There's random factor. An output could be
(1,.,.) =
-0.7684499      -0.49320614     
-0.98002595     -0.47857404     

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
grad_output = np.array([[
  [2.0, 3.0],
  [4.0, 5.0]
]])
rec = Recurrent()

model = Sequential() \
    .add(rec.add(GRU(input_size, hidden_size))) \
    .add(TimeDistributed(Linear(hidden_size, output_size)))
output = model.forward(input)
gradient = model.backward(input, grad_output)
-> print output
# There's random factor. An output could be
[[[ 0.27857888  0.20263115]
  [ 0.29470384  0.22594413]]]
-> print gradient
[[[-0.32956457  0.27405274]
  [-0.32718879  0.32963118]]]
```
---
## ConvLSTMPeephole ##

**Scala:**
```scala
val model = ConvLSTMPeephole(
  inputSize = 2,
  outputSize = 4,
  kernelI = 3,
  kernelC = 3,
  stride = 1,
  wRegularizer = null,
  uRegularizer = null,
  bRegularizer = null,
  cRegularizer = null,
  withPeephole = true)
```

**Python:**
```python
model = ConvLSTMPeephole(
  input_size = 2,
  output_size = 4,
  kernel_i = 3,
  kernel_c = 3,
  stride = 1,
  wRegularizer=None,
  uRegularizer=None,
  bRegularizer=None,
  cRegularizer = None,
  with_peephole = True)
```

Convolution Long Short Term Memory architecture with peephole for 2 dimension images.
The input tensor in `forward(input)` is expected to be a 5D tensor (`batch x time x nInputPlane x height x width`). output of
`forward(input)` is also expected to be a 5D tensor (`batch x time x outputPlane x height x width`).

Ref.

1. https://arxiv.org/abs/1506.04214 (blueprint for this module)
2. https://github.com/viorik/ConvLSTM

Parameters:

* `inputSize` number of input planes in the image given into forward()
* `outputSize` number of output planes the convolution layer will produce
* `kernelI` convolutional filter size to convolve input
* `kernelC` convolutional filter size to convolve cell
* `stride` step of the convolution
* `wRegularizer` instance of [[Regularizer]]
                   (eg. L1 or L2 regularization), applied to the input weights matrices.
* `uRegularizer` instance [[Regularizer]]
          (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
* `bRegularizer` instance of [[Regularizer]]
          applied to the bias.
* `cRegularizer` instance of [[Regularizer]]
        applied to peephole.
* `withPeephole` whether use last cell status control a gate

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

val outputSize = 4
val inputSize = 3
val seqLength = 2
val batchSize = 1
               
val input = Tensor(Array(batchSize, seqLength, inputSize, 3, 3)).rand()

val rec = Recurrent()
    val model = Sequential()
      .add(rec
        .add(ConvLSTMPeephole(inputSize, outputSize, 3, 3, 1, withPeephole = false)))
        
val output = model.forward(input).toTensor

scala> print(input)
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
0.32810056      0.23436882      0.1387327
0.98273766      0.76427716      0.73554766
0.47947738      0.72805804      0.43982902

(1,1,2,.,.) =
0.58144385      0.7534736       0.94412255
0.05087549      0.021427812     0.91333073
0.6844351       0.62977004      0.68027127

(1,1,3,.,.) =
0.48504198      0.16233416      0.7612549
0.5387952       0.8391377       0.3687795
0.85271466      0.71726906      0.79466575

(1,2,1,.,.) =
0.727532        0.05341824      0.32531977
0.79593664      0.60162276      0.99931896
0.7534103       0.71214366      0.031062916

(1,2,2,.,.) =
0.7343414       0.053005006     0.7448063
0.2277985       0.47414783      0.21945253
0.0034818714    0.11545401      0.73085403

(1,2,3,.,.) =
0.9644807       0.30755267      0.42099005
0.6831594       0.50683653      0.14237563
0.65172654      0.86954886      0.5077393

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x2x3x3x3]

scala> print(output)
(1,1,1,.,.) =
-0.04460164     -0.023752786    -0.014343993
0.0067705153    0.08542874      0.020885356
-0.042719357    -0.012113815    -0.030324051

(1,1,2,.,.) =
-0.038318213    -0.056998547    -0.02303889
0.027873239     -0.040311974    -0.03261278
0.015056128     0.11064132      0.0034682436

(1,1,3,.,.) =
0.006952648     0.011758738     -0.047590334
0.052022297     0.040250845     -0.046224136
-0.0084472215   -0.02629062     -0.0737972

(1,1,4,.,.) =
-0.087721705    0.0382758       0.027436329
-0.030658737    -0.022953996    0.15838619
0.055106055     0.004877564     0.098199464

(1,2,1,.,.) =
-0.069991425    -0.022071177    -0.06291955
-0.006841902    0.010781053     0.05410414
-0.03933395     -0.003422904    -0.106903486

(1,2,2,.,.) =
-0.059429795    -0.098534085    -0.068920344
0.008100101     0.01948546      -0.040567685
0.048763007     0.06001041      0.003068042

(1,2,3,.,.) =
0.02817994      0.006684172     -0.0962587
0.022453573     0.014425971     -0.06118475
-0.013392928    -0.04574135     -0.12722406

(1,2,4,.,.) =
-0.074006446    -0.028510522    0.06808455
-0.021926142    0.036675904     0.18708621
0.08240187      0.12469789      0.17341805

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x4x3x3]
```
**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

output_size = 4
input_size= 3
seq_len = 2
batch_size = 1
               
input = np.random.randn(batch_size, seq_len, input_size, 3, 3)
rec = Recurrent()
model = Sequential().add(
    rec.add(ConvLSTMPeephole(input_size, output_size, 3, 3, 1, with_peephole = False)))
output = model.forward(input)

>>> print(input)
[[[[[ 2.39979422  0.75647109  0.88928214]
    [-0.07132477 -0.4348564   0.38270011]
    [-1.03522309  0.38399781  0.20369625]]

   [[-0.48392771  0.54371842 -1.42064221]
    [-0.3711481  -0.16019682  0.82116693]
    [ 0.15922215  1.79676148  0.38362552]]

   [[-0.69402482  1.11930766 -1.29138064]
    [ 0.92755002 -0.31138235  0.34953374]
    [-0.0176643   1.13839126  0.02133309]]]


  [[[-0.40704988  0.1819258  -0.21400335]
    [ 0.65717965  0.75912824  1.49077775]
    [-0.74917913 -1.48460681  1.06098727]]

   [[ 1.04942415  1.2558929  -1.24367776]
    [-0.13452707  0.01485188  2.41215047]
    [ 0.59776321 -0.38602613  0.57937933]]

   [[ 0.55007301  1.22571134  0.11656841]
    [-0.4722457   1.79801493  0.59698431]
    [ 0.25119458 -0.27323404  1.5516505 ]]]]]
  
>>> print(output)
[[[[[-0.22908808 -0.08243818 -0.10530333]
    [ 0.04545299  0.0347576   0.06448466]
    [ 0.00148075 -0.01422587 -0.04424585]]

   [[-0.08625289  0.00121372  0.00961097]
    [-0.08068027  0.2389598  -0.08875058]
    [-0.10860988 -0.08109165  0.05274875]]

   [[ 0.01545026 -0.14079301  0.0162897 ]
    [ 0.0114354   0.01696588  0.09375648]
    [ 0.06766916  0.16015787 -0.01530124]]

   [[-0.00311095  0.07033439  0.05258823]
    [-0.04846094 -0.11335927 -0.22434352]
    [-0.09923813 -0.064981   -0.05341392]]]


  [[[-0.01070079  0.01705431 -0.10199456]
    [-0.19023973 -0.1359819   0.11552753]
    [ 0.04331793  0.00603994 -0.19059387]]

   [[-0.12100818 -0.01191896  0.08049219]
    [-0.10134248  0.02910084 -0.00024394]
    [-0.09548382 -0.18623565  0.18261637]]

   [[-0.00644266  0.03494127  0.09105418]
    [ 0.03467004 -0.1236406   0.23844369]
    [ 0.12281432  0.09469442  0.04526915]]

   [[ 0.00190313  0.01997324 -0.17609949]
    [-0.0937     -0.03763293 -0.04860835]
    [-0.15700462 -0.17341313 -0.06551415]]]]]
```

---
## ConvLSTMPeephole3D ##

**Scala:**
```scala
val model = ConvLSTMPeephole3D(
  inputSize = 2,
  outputSize = 4,
  kernelI = 3,
  kernelC = 3,
  stride = 1,
  wRegularizer = null,
  uRegularizer = null,
  bRegularizer = null,
  cRegularizer = null,
  withPeephole = true)
```

**Python:**
```python
model = ConvLSTMPeephole3D(
  input_size = 2,
  output_size = 4,
  kernel_i = 3,
  kernel_c = 3,
  stride = 1,
  wRegularizer=None,
  uRegularizer=None,
  bRegularizer=None,
  cRegularizer=None,
  with_peephole = True)
```

Similar to Convlstm2D, it's a Convolution Long Short Term Memory architecture with peephole but for 3 spatial dimension images.
The input tensor in `forward(input)` is expected to be a 6D tensor (`batch x time x nInputPlane x height x width x length`). output of
`forward(input)` is also expected to be a 6D tensor (`batch x time x outputPlane x height x width x length`).

Parameters:

* `inputSize` number of input planes in the image given into forward()
* `outputSize` number of output planes the convolution layer will produce
* `kernelI` convolutional filter size to convolve input
* `kernelC` convolutional filter size to convolve cell
* `stride` step of the convolution
* `wRegularizer` instance of [[Regularizer]]
                   (eg. L1 or L2 regularization), applied to the input weights matrices.
* `uRegularizer` instance [[Regularizer]]
          (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
* `bRegularizer` instance of [[Regularizer]]
          applied to the bias.
* `cRegularizer` instance of [[Regularizer]]
          applied to peephole.
* `withPeephole` whether use last cell status control a gate

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

val outputSize = 4
val inputSize = 3
val seqLength = 2
val batchSize = 1
               
val input = Tensor(Array(batchSize, seqLength, inputSize, 3, 3, 3)).rand()

val rec = Recurrent()
    val model = Sequential()
      .add(rec
        .add(ConvLSTMPeephole3D(inputSize, outputSize, 3, 3, 1, withPeephole = false)))
        
val output = model.forward(input).toTensor

scala> print(input)
(1,1,1,1,.,.) =
0.42592695	0.32742274	0.7926296	
0.21923159	0.7427106	0.31764257	
0.121872835	0.54231954	0.32091624	

(1,1,1,2,.,.) =
0.06762145	0.8054027	0.8297814	
0.95535785	0.20807801	0.46387103	
0.90996957	0.7849159	0.79179865	

(1,1,1,3,.,.) =
0.22927228	0.29869995	0.1145133	
0.12646529	0.8917339	0.7545332	
0.8044227	0.5340327	0.9784876	

(1,1,2,1,.,.) =
0.68444395	0.47932255	0.28224406	
0.5083046	0.9364489	0.27006733	
0.24699332	0.55712855	0.50037974	

(1,1,2,2,.,.) =
0.46334672	0.10979338	0.6378528	
0.8557069	0.10780747	0.73767877	
0.12505454	0.72492164	0.5440267	

(1,1,2,3,.,.) =
0.15598479	0.52033675	0.64091414	
0.15149859	0.64515823	0.6023936	
0.31461328	0.1901752	0.98015004	

(1,1,3,1,.,.) =
0.9700778	0.24109624	0.23764393	
0.16602103	0.97310185	0.072756775	
0.849201	0.825025	0.2753475	

(1,1,3,2,.,.) =
0.8621034	0.24596989	0.56645423	
0.004375741	0.9873366	0.89219636	
0.56948274	0.291723	0.5503815	

(1,1,3,3,.,.) =
0.626368	0.9389012	0.8974684	
0.8553843	0.39709046	0.372683	
0.38087663	0.94703597	0.71530545	

(1,2,1,1,.,.) =
0.74050623	0.39862877	0.57509166	
0.87832487	0.41345102	0.6262451	
0.665165	0.49570015	0.8304163	

(1,2,1,2,.,.) =
0.30847755	0.51876235	0.10555197	
0.10103849	0.9479695	0.11847988	
0.60081536	0.003097216	0.22800316	

(1,2,1,3,.,.) =
0.113101795	0.76638913	0.091707565	
0.30347276	0.029687135	0.37973404	
0.67719024	0.02180517	0.12747364	

(1,2,2,1,.,.) =
0.12513511	0.74210113	0.82569206	
0.1406212	0.7400157	0.041633762	
0.26903376	0.6195371	0.618376	

(1,2,2,2,.,.) =
0.068732955	0.09746146	0.15479624	
0.57418007	0.7181547	0.6494809	
0.29213288	0.35022008	0.15421997	

(1,2,2,3,.,.) =
0.47196773	0.55650383	0.938309	
0.70717365	0.68351734	0.32646814	
0.99775004	0.2596666	0.6803594	

(1,2,3,1,.,.) =
0.6320722	0.105437785	0.36752152	
0.8347324	0.38376364	0.641918	
0.40254018	0.5421287	0.792421	

(1,2,3,2,.,.) =
0.2652298	0.6261154	0.21971565	
0.31418183	0.44987184	0.43880364	
0.76821107	0.17070894	0.47295105	

(1,2,3,3,.,.) =
0.16514553	0.37016368	0.23397927	
0.19776458	0.07518195	0.48995376	
0.13584352	0.23562871	0.41726747	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x2x3x3x3x3]

scala> print(output)
(1,1,1,1,.,.) =
0.014528348	0.03160259	0.05313618	
-0.011796958	0.027994404	0.028153816	
-0.010374474	0.029486801	0.033610236	

(1,1,1,2,.,.) =
0.07966786	0.041255455	0.09181337	
0.025984935	0.06594588	0.07572434	
0.019637575	0.0068716113	0.03775029	

(1,1,1,3,.,.) =
0.07043511	0.044567406	0.08229201	
0.10589862	0.109124646	0.0888148	
0.018544039	0.04097363	0.09130414	

(1,1,2,1,.,.) =
0.1032162	-0.01981514	-0.0016546922	
0.026028564	0.0100736385	0.009424217	
-0.048695907	-0.009172593	-0.029458746	

(1,1,2,2,.,.) =
0.058081806	0.101963215	0.056670886	
0.09300327	0.035424378	0.02410931	
0.056604195	-0.0032351227	0.027961217	

(1,1,2,3,.,.) =
0.11710516	0.09371774	-0.013825272	
0.02930173	0.06391968	0.04034334	
0.010447707	-0.004905071	0.011929871	

(1,1,3,1,.,.) =
-0.020980358	0.08554982	-0.07644813	
0.06367171	-0.06037125	0.019925931	
0.0026421212	0.051610045	0.023478134	

(1,1,3,2,.,.) =
-0.033074334	-0.0381583	-0.019341394	
-0.0625153	-0.06907081	-0.019746307	
-0.010362335	0.0062695937	0.054116223	

(1,1,3,3,.,.) =
0.00461099	-0.03308314	-6.8137434E-4	
-0.075023845	-0.024970314	0.008133534	
0.019836657	0.051302493	0.043689556	

(1,1,4,1,.,.) =
0.027088374	0.008537832	-0.020948375	
0.021569671	0.016515112	-0.019221392	
-0.0074050943	-0.03274501	0.003256779	

(1,1,4,2,.,.) =
8.967657E-4	0.019020535	-0.05990117	
0.06226491	-0.017516658	-0.028854925	
0.048010994	0.031080479	-4.8373322E-4	

(1,1,4,3,.,.) =
0.03253352	-0.023469497	-0.047273926	
-0.03765316	0.011091222	0.0036612307	
0.050733108	0.01736545	0.0061482657	

(1,2,1,1,.,.) =
-0.0037416879	0.03895818	0.102294624	
0.011019588	0.03201482	0.07654998	
-0.015550408	0.009587483	0.027655594	

(1,2,1,2,.,.) =
0.089279816	0.03306113	0.11713534	
0.07299529	0.057692382	0.11090511	
-0.0031341386	0.091527686	0.07210587	

(1,2,1,3,.,.) =
0.080724075	0.07707712	0.07624206	
0.06552311	0.104010254	0.09213451	
0.07030998	0.0022800618	0.12461836	

(1,2,2,1,.,.) =
0.10180804	0.020320226	-0.0025817656	
0.016294254	-0.024293585	-0.004399727	
-0.032854877	1.1120379E-4	-0.02109197	

(1,2,2,2,.,.) =
0.0968586	0.07098973	0.07648221	
0.0918679	0.10268471	0.056947876	
0.027774762	-0.03927014	0.04663368	

(1,2,2,3,.,.) =
0.10225944	0.08460646	-8.393754E-4	
0.051307157	0.011988232	0.037762236	
0.029469138	0.023369621	0.037675448	

(1,2,3,1,.,.) =
-0.017874755	0.08561468	-0.066132575	
0.010558257	-0.01448278	0.0073027355	
-0.007930762	0.052643955	0.008378773	

(1,2,3,2,.,.) =
-0.009250246	-0.06543376	-0.025082456	
-0.093004115	-0.08637037	-0.063408665	
-0.06941878	0.010163672	0.07595171	

(1,2,3,3,.,.) =
0.014756428	-0.040423956	-0.011537984	
-0.046337806	-0.008416044	0.068246834	
3.5782385E-4	0.056929104	0.052956138	

(1,2,4,1,.,.) =
0.033539586	0.013915413	-0.024538055	
0.042590756	0.034134552	0.021031722	
-0.026687687	0.0012957935	-0.0053077694	

(1,2,4,2,.,.) =
0.0033482902	-0.037335612	-0.0956953	
0.007350738	-0.05237038	-0.08849126	
0.016356941	0.032067236	-0.0012172575	

(1,2,4,3,.,.) =
-0.020006038	-0.030038685	-0.054900024	
-0.014171911	0.01270077	-0.004130667	
0.04607582	0.040028486	0.011846061	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x4x3x3x3]

```
**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

output_size = 4
input_size= 3
seq_len = 2
batch_size = 1
               
input = np.random.randn(batch_size, seq_len, input_size, 3, 3, 3)
rec = Recurrent()
model = Sequential().add(
    rec.add(ConvLSTMPeephole3D(input_size, output_size, 3, 3, 1, with_peephole = False)))
output = model.forward(input)

>>> print(input)
[[[[[[ -8.92954769e-02  -9.77685543e-03   1.97566296e+00]
     [ -5.76910662e-01  -9.08404346e-01  -4.70799006e-01]
     [ -9.86229768e-01   7.87303916e-01   2.29691167e+00]]

    [[ -7.48240036e-01   4.12766483e-01  -3.88947296e-01]
     [ -1.39879028e+00   2.43984720e+00  -2.43947000e-01]
     [  1.86468980e-01   1.34599111e+00  -6.97932324e-01]]

    [[  1.23278710e+00  -4.02661913e-01   8.50721265e-01]
     [ -1.79452089e-01  -5.58813385e-01   1.10060751e+00]
     [ -6.27181580e-01  -2.69531726e-01  -1.07857962e-01]]]


   [[[ -1.01462355e+00   5.47520811e-02   3.06976674e-01]
     [  9.64871158e-01  -1.16953916e+00   1.41880629e+00]
     [  1.19127007e+00   1.71403439e-01  -1.30787798e+00]]

    [[ -6.44313121e-01  -8.45131087e-01   6.99275525e-02]
     [ -3.07656855e-01   1.25746926e+00   3.89980508e-02]
     [ -2.59853355e-01   8.78915612e-01  -9.37204072e-02]]

    [[  7.69958423e-02  -3.22523203e-01  -7.31295167e-01]
     [  1.46184856e+00   1.88641278e+00   1.46645372e-01]
     [  4.38390570e-01  -2.85102515e-01  -1.81269541e+00]]]


   [[[  2.95126419e-01  -1.13715815e+00   9.36848777e-01]
     [ -1.62071909e+00  -1.06018926e+00   1.88416944e+00]
     [ -5.81248254e-01   1.05162543e+00  -3.58790528e-01]]

    [[ -7.54710826e-01   2.29994522e+00   7.24276828e-01]
     [  5.77031441e-01   7.36132125e-01   2.24719266e+00]
     [ -4.53710071e-05   1.98478259e-01  -2.62825655e-01]]

    [[  1.68124733e+00  -9.97417864e-01  -3.73490116e-01]
     [ -1.12558844e+00   2.60032255e-01   9.67994680e-01]
     [  1.78486852e+00   1.17514142e+00  -1.96871551e-01]]]]



  [[[[  4.43156770e-01  -4.42279658e-01   8.00893010e-01]
     [ -2.04817319e-01  -3.89658940e-01  -1.10950351e+00]
     [  6.61008455e-01  -4.07251176e-01   1.14871901e+00]]

    [[ -2.07785815e-01  -8.92450022e-01  -4.23830113e-02]
     [ -5.26555807e-01   3.76671145e-02  -2.17877979e-01]
     [ -7.68371469e-01   1.53052409e-01   1.02405949e+00]]

    [[  5.75018628e-01  -9.47162716e-01   6.47917376e-01]
     [  4.66967303e-01   1.00917068e-01  -1.60894238e+00]
     [ -1.46491032e-01   3.17782758e+00   1.12581079e-01]]]


   [[[  9.32343396e-01  -1.03853742e+00   5.67577254e-02]
     [  1.25266813e+00   3.52463164e-01  -1.86783652e-01]
     [ -1.20321270e+00   3.95144053e-01   2.09975625e-01]]

    [[  2.68240844e-01  -1.34931544e+00   1.34259455e+00]
     [  6.34339337e-01  -5.21231073e-02  -3.91895492e-01]
     [  1.53872699e-01  -5.07236962e-02  -2.90772390e-01]]

    [[ -5.07933749e-01   3.78036493e-01   7.41781186e-01]
     [  1.62736825e+00   1.24125644e+00  -3.97490478e-01]
     [  5.77762257e-01   1.10372911e+00   1.58060183e-01]]]


   [[[  5.31859839e-01   1.72805654e+00  -3.77124271e-01]
     [  1.24638369e+00  -1.54061928e+00   6.22001793e-01]
     [  1.92447446e+00   7.71351435e-01  -1.59998400e+00]]

    [[  1.44289958e+00   5.41433535e-01   9.19769038e-01]
     [  9.92873720e-01  -9.05746035e-01   1.35906705e+00]
     [  1.38994943e+00   2.11451648e+00  -1.58783119e-01]]

    [[ -1.44024889e+00  -5.12269041e-01   8.56761529e-02]
     [  1.16668889e+00   7.58164067e-01  -1.04304927e+00]
     [  6.34138215e-01  -7.89939971e-01  -5.52376307e-01]]]]]]

>>> print(output)
[[[[[[ 0.08801123 -0.15533912 -0.08897342]
     [ 0.01158205 -0.01103314  0.02793931]
     [-0.01269898 -0.09544773  0.03573112]]

    [[-0.15603164 -0.16063154 -0.09672774]
     [ 0.15531734  0.05808824 -0.01653268]
     [-0.06348733 -0.10497692 -0.13086422]]

    [[ 0.002062   -0.01604773 -0.14802884]
     [-0.0934701  -0.06831796  0.07375477]
     [-0.01157693  0.17962074  0.13433206]]]


   [[[ 0.03571969 -0.20905718 -0.05286504]
     [-0.18766534 -0.10728011  0.04605131]
     [-0.07477143  0.02631984  0.02496208]]

    [[ 0.06653454  0.06536704  0.01587131]
     [-0.00348636 -0.04439256  0.12680793]
     [ 0.00328905  0.01904229 -0.06607334]]

    [[-0.04666118 -0.06754828  0.07643934]
     [-0.05434367 -0.09878142  0.06385987]
     [ 0.02643086 -0.01466259 -0.1031612 ]]]


   [[[-0.0572568   0.13133277 -0.0435285 ]
     [-0.11612531  0.09036689 -0.09608591]
     [-0.01049453 -0.02091818 -0.00642477]]

    [[ 0.1255362  -0.07545673 -0.07554446]
     [ 0.07270454 -0.24932131 -0.13024282]
     [ 0.05507039 -0.0109083   0.00408967]]

    [[-0.1099453  -0.11417828  0.06235902]
     [ 0.03701246 -0.02138007 -0.05719795]
     [-0.02627739 -0.15853535 -0.01103899]]]


   [[[ 0.10380347 -0.05826453 -0.00690799]
     [ 0.01000955 -0.11808137 -0.039118  ]
     [ 0.02591963 -0.03464907 -0.21320052]]

    [[-0.03449376 -0.00601143  0.05562805]
     [ 0.09242225  0.01035819  0.09432289]
     [-0.12854564  0.189775   -0.06698175]]

    [[ 0.03462109  0.02545513 -0.14716192]
     [ 0.02003146 -0.03616474  0.04574323]
     [ 0.04782774 -0.04594192  0.01773669]]]]



  [[[[ 0.04205685 -0.05454008 -0.0389443 ]
     [ 0.07172828  0.03370164  0.00703573]
     [ 0.01299563 -0.06371058  0.02505058]]

    [[-0.09191396  0.06227853 -0.15412274]
     [ 0.09069916  0.01907965 -0.05783302]
     [-0.03441796 -0.11438221 -0.1011953 ]]

    [[-0.00837748 -0.06554071 -0.14735688]
     [-0.04640726  0.01484136  0.14445931]
     [-0.09255736 -0.12196805 -0.0444463 ]]]


   [[[ 0.01632853  0.01925437  0.02539274]
     [-0.09239745 -0.13713452  0.06149488]
     [-0.01742462  0.06624916  0.01490385]]

    [[ 0.03866836  0.19375585  0.06069621]
     [-0.11291414 -0.29582706  0.11678439]
     [-0.09451667  0.05238266 -0.05152772]]

    [[-0.11206269  0.09128021  0.09243178]
     [ 0.01127258 -0.05845089  0.09795895]
     [ 0.00747248  0.02055444  0.0121724 ]]]


   [[[-0.11144694 -0.0030012  -0.03507657]
     [-0.15461211 -0.00992483  0.02500556]
     [-0.07733752 -0.09037463  0.02955181]]

    [[-0.00988597  0.0264726  -0.14286363]
     [-0.06936073 -0.01345975 -0.16290392]
     [-0.07821255 -0.02489748  0.05186536]]

    [[-0.12142604  0.04658077  0.00509979]
     [-0.16115788 -0.19458961 -0.04082467]
     [ 0.10544231 -0.10425973  0.01532217]]]


   [[[ 0.08169251  0.05370622  0.00506061]
     [ 0.08195242  0.08890768  0.03178475]
     [-0.03648232  0.02655745 -0.18274172]]

    [[ 0.07358464 -0.09604233  0.06556321]
     [-0.02229194  0.17364709  0.07240117]
     [-0.18307404  0.04115544 -0.15400645]]

    [[ 0.0156146  -0.15857749 -0.12837477]
     [ 0.07957774  0.06684072  0.0719762 ]
     [-0.13781127 -0.03935293 -0.096707  ]]]]]]

```

---
## TimeDistributed ##

**Scala:**
```scala
val layer = TimeDistributed(layer)
```
**Python:**
```python
layer = TimeDistributed(layer)
```

This layer is intended to apply contained layer to each temporal time slice
of input tensor.

The input data format is [Batch, Time, Other dims]. For the contained layer, it must not change
the Other dims length.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val layer = TimeDistributed(Sum(1, squeeze = false, nInputDims = 2))
val input = Tensor(T(T(
  T(
    T(1.0f, 2.0f),
    T(3.0f, 4.0f)
  ),
  T(
    T(2.0f, 3.0f),
    T(4.0f, 5.0f)
  )
)))
layer.forward(input)
layer.backward(input, Tensor(T(T(
  T(
    T(0.1f, 0.2f)
  ),
  T(
    T(0.3f, 0.4f)
  )
))))
```
Gives the output,
```
(1,1,.,.) =
4.0     6.0

(1,2,.,.) =
6.0     8.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x1x2]

(1,1,.,.) =
0.1     0.2
0.1     0.2

(1,2,.,.) =
0.3     0.4
0.3     0.4

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x2x2]

```

**Python example:**
```python
from bigdl.nn.layer import TimeDistributed,Sum
import numpy as np

layer = TimeDistributed(Sum(1, squeeze = False, n_input_dims = 2))

input = np.array([[
  [
    [1.0, 2.0],
    [3.0, 4.0]
  ],
  [
    [2.0, 3.0],
    [4.0, 5.0]
  ]
]])
layer.forward(input)
layer.backward(input, np.array([[
  [
    [0.1, 0.2]
  ],
  [
    [0.3, 0.4]
  ]
]]))
```
Gives the output,
```
array([[[[ 4.,  6.]],

        [[ 6.,  8.]]]], dtype=float32)
        
array([[[[ 0.1       ,  0.2       ],
         [ 0.1       ,  0.2       ]],

        [[ 0.30000001,  0.40000001],
         [ 0.30000001,  0.40000001]]]], dtype=float32)
```

