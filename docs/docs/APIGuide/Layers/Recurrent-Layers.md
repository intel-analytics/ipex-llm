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

Recurrent supports returning state and cell of its rnn cells at last time step by using getHiddenState. output of getHiddenState
is an Activity.

If contained cell is simple rnn, getHiddenState return value is a tensor(hidden state) which is `batch x hiddenSize`.  
If contained cell is lstm, getHiddenState return value is a table [hidden state, cell], both size is `batch x hiddenSize`.  
If contained cell is convlstm, getHiddenState return value is a table [hidden state, cell], both size is `batch x outputPlane x height x width`.  
If contained cell is convlstm3D, getHiddenState return value is a table [hidden state, cell], both size is `batch x outputPlane x height x width x length`.

Recurrent also support init hidden state by using setHiddenState, currently only scala version. After we get hidden state from getHiddenState, we can directly used it in setHiddenState, which will set hidden state and cell at the first time step.

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

val state = module.getHiddenState()
module.setHiddenState(state)

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

res = module.get_hidden_state()

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

## RecurrentDecoder ##

**Scala:**
```scala
val module = RecurrentDecoder(outputLength = 5)
```
**Python:**
```python
module = RecurrentDecoder(output_length = 5)
```

RecurrentDecoder module is a container of rnn cells which used to make
a prediction of the next timestep based on the prediction we made from
the previous timestep.

Input for RecurrentDecoder has to be batch x stepShape(shape of the input at a single time step). 

During training, input at t(i) is output at t(i-1), input at t(0) is
user input.

Output for RecurrentDecoder has to be batch x outputLen x shape.
 
With RecurrentDecoder, inputsize and hiddensize of the cell must be the same.

Different types of rnn cells can be added using add() function.

Parameters:

* `outputLength` sequence length of output

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val hiddenSize = 4
val inputSize = 4
val batchSize = 2
val module = RecurrentDecoder(5).add(LSTMPeephole(inputSize, hiddenSize))
val input = Tensor(Array(batchSize, inputSize)).rand()

val output = module.forward(input)

> input
0.32985476	0.5081215	0.95177317	0.24744023	
0.030384725	0.4868633	0.7781735	0.8046177	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x4]

> output
(1,.,.) =
-0.055717956	-0.14357334	0.011429226	0.10056843	
-0.013699859	-0.078585915	0.050289743	0.027037282	
0.011173044	-0.07941696	0.07381668	0.0020067326	
0.016142089	-0.081511036	0.08775896	-0.011746041	
0.0149942655	-0.08317861	0.09522702	-0.018894192	

(2,.,.) =
-0.041173447	-0.10931831	-0.04198869	0.1287807	
0.010115819	-0.07071178	0.011613955	0.04737701	
0.027745798	-0.07493171	0.054053202	0.010752724	
0.02633817	-0.07929653	0.07783712	-0.008406129	
0.020732995	-0.08214355	0.09030104	-0.017894702	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x5x4]


```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

hidden_size = 4
input_size = 4
batch_size = 2
module = RecurrentDecoder(5).add(LSTMPeephole(input_size, hidden_size))
input = np.random.randn(batch_size, input_size)

output = module.forward(input)

> input
[[ 0.81779139 -0.55165689 -1.5898894   0.03572801]
 [ 0.77645041 -0.39702404  0.16826132  1.37081681]]

> output
[[[ 0.0492445  -0.26821002 -0.13461511  0.13712646]
  [ 0.11038809 -0.22399209 -0.15706871  0.17625453]
  [ 0.12579349 -0.20708388 -0.17392202  0.19129401]
  [ 0.12953098 -0.20042329 -0.1886536   0.20086248]
  [ 0.12905654 -0.19860952 -0.19987412  0.20697045]]

 [[ 0.146652   -0.12099689  0.05711044  0.03263233]
  [ 0.15229702 -0.12689863 -0.05258115  0.09761411]
  [ 0.14552552 -0.13706802 -0.11870711  0.13544162]
  [ 0.13672781 -0.15158641 -0.16068494  0.16216366]
  [ 0.13007095 -0.16579619 -0.18658556  0.18039529]]]
```
---
## RNN ##

**Scala:**
```scala
val rnnCell = RnnCell[Double](
  inputSize,
  hiddenSize,
  activation,
  isInputWithBias = true,
  isHiddenWithBias = true,
  wRegularizer = null,
  uRegularizer = null,
  bRegularizer = null)
```
**Python:**
```python
rnnCell = RnnCell(
  input_size,
  hidden_size,
  activation,
  isInputWithBias=True,
  isHiddenWithBias=True,
  wRegularizer=None,
  uRegularizer=None,
  bRegularizer=None)
```

Implementation of vanilla recurrent neural network cell

The input tensor in `forward(input)` is expected to be a 3D tensor (`batch x time x inputSize`). output of
`forward(input)` is also expected to be a 3D tensor (`batch x time x hiddenSize`).

The updating is defined as:

```
h_t = f(i2h * x_t + h2h * h_{t-1})
```
where
* `i2h` weight matrix of input to hidden units
* `h2h` weight matrix of hidden units to themselves through time

Parameters:

* `inputSize` input size. Default: 4
* `hiddenSize`  hidden layer size. Default: 3
* `activation` instance of activation function for non-linearity.
  For Python, one can also pass the name of an existing activation as a string, eg. 'tanh', 'relu', 'sigmoid', 'hard_sigmoid', 'softmax' etc.
* `isInputWithBias` boolean, whether to contain bias for input. Default: true
* `isHiddenWithBias` boolean, whether to contain bias for hidden layer. Default: true
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
val lstm = LSTM(
  inputSize,
  hiddenSize,
  p = 0.0,
  activation = null,
  innerActivation = null,
  wRegularizer = null,
  uRegularizer = null,
  bRegularizer = null)
```
**Python:**
```python
lstm = LSTM(
  input_size,
  hidden_size,
  p=0.0,
  activation=None,
  inner_activation=None,
  wRegularizer=None,
  uRegularizer=None,
  bRegularizer=None)
```

Long Short Term Memory architecture.
The input tensor in `forward(input)` is expected to be a 3D tensor (`batch x time x inputSize`). output of
`forward(input)` is also expected to be a 3D tensor (`batch x time x hiddenSize`).

Ref:

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
* `activation` activation function, by default to be `Tanh` if not specified.
  For Python, one can also pass the name of an existing activation as a string, eg. 'tanh', 'relu', 'sigmoid', 'hard_sigmoid', 'softmax' etc.
* `innerActivation` activation function for inner cells, by default to be `Sigmoid` if not specified.
  For Python, one can also pass the name of an existing activation as a string, eg. 'tanh', 'relu', 'sigmoid', 'hard_sigmoid', 'softmax' etc.
* `wRegularizer` instance of [[Regularizer]] (eg. L1 or L2 regularization), applied to the input weights matrices.
* `uRegularizer` instance of [[Regularizer]] (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
* `bRegularizer` instance of [[Regularizer]], applied to the bias.

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
The input tensor in `forward(input)` is expected to be a 3D tensor (`batch x time x inputSize`). output of
`forward(input)` is also expected to be a 3D tensor (`batch x time x hiddenSize`).

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
* `wRegularizer` instance of [[Regularizer]] (eg. L1 or L2 regularization), applied to the input weights matrices.
* `uRegularizer` instance of [[Regularizer]] (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
* `bRegularizer` instance of [[Regularizer]], applied to the bias.

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
val gru = GRU(
  inputSize,
  outputSize,
  p = 0.0,
  activation = null,
  innerActivation = null,
  wRegularizer = null,
  uRegularizer = null,
  bRegularizer = null)
```
**Python:**
```python
gru = GRU(
  inputSize,
  outputSize,
  p=0.0,
  activation=None,
  inner_activation=None,
  wRegularizer=None,
  uRegularizer=None,
  bRegularizer=None)
```

Gated Recurrent Units architecture. The first input in sequence uses zero value for cell and hidden state.
The input tensor in `forward(input)` is expected to be a 3D tensor (`batch x time x inputSize`). output of
`forward(input)` is also expected to be a 3D tensor (`batch x time x outputSize`).

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
* `activation` activation function, by default to be `Tanh` if not specified.
  For Python, one can also pass the name of an existing activation as a string, eg. 'tanh', 'relu', 'sigmoid', 'hard_sigmoid', 'softmax' etc.
* `innerActivation` activation function for inner cells, by default to be `Sigmoid` if not specified.
  For Python, one can also pass the name of an existing activation as a string, eg. 'tanh', 'relu', 'sigmoid', 'hard_sigmoid', 'softmax' etc.
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
  padding = -1,
  activation = null,
  innerActivation = null,
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
  padding = -1,
  activation = None,
  inner_activation = None,
  wRegularizer = None,
  uRegularizer = None,
  bRegularizer = None,
  cRegularizer = None,
  with_peephole = True)
```

Convolution Long Short Term Memory architecture with peephole for 2 dimension images.
The input tensor in `forward(input)` is expected to be a 4D or 5D tensor
If ConvLSTM work with Recurrent, input is 5D tensor (`batch x time x nInputPlane x height x width`). output of
`forward(input)` is also expected to be a 5D tensor (`batch x time x outputPlane x height x width`).

If ConvLSTM work with RecurrentDecoder, input is 4D tensor (`batch x nInputPlane x height x width`). output of
`forward(input)` is expected to be a 5D tensor (`batch x outputLen x outputPlane x height x width`).

Ref.

1. https://arxiv.org/abs/1506.04214 (blueprint for this module)
2. https://github.com/viorik/ConvLSTM

Parameters:

* `inputSize` number of input planes in the image given into forward()
* `outputSize` number of output planes the convolution layer will produce
* `kernelI` convolutional filter size to convolve input
* `kernelC` convolutional filter size to convolve cell
* `stride` step of the convolution, default is 1
* `padding` step of the convolution, default is -1, behaves same with SAME padding in tensorflow
                 Default stride,padding value ensure last 2 dim of output shape is the same with input
* `activation` activation function, by default to be `Tanh` if not specified.
  For Python, one can also pass the name of an existing activation as a string, eg. 'tanh', 'relu', 'sigmoid', 'hard_sigmoid', 'softmax' etc.
* `innerActivation` activation function for inner cells, by default to be `Sigmoid` if not specified.
  For Python, one can also pass the name of an existing activation as a string, eg. 'tanh', 'relu', 'sigmoid', 'hard_sigmoid', 'softmax' etc.
* `wRegularizer` instance of [[Regularizer]] (eg. L1 or L2 regularization), applied to the input weights matrices.
* `uRegularizer` instance of [[Regularizer]] (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
* `bRegularizer` instance of [[Regularizer]], applied to the bias.
* `cRegularizer` instance of [[Regularizer]], applied to peephole.
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
  padding = -1,
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
  padding = -1,
  wRegularizer=None,
  uRegularizer=None,
  bRegularizer=None,
  cRegularizer=None,
  with_peephole = True)
```

Similar to Convlstm2D, it's a Convolution Long Short Term Memory architecture with peephole but for 3 spatial dimension images.
The input tensor in `forward(input)` is expected to be a 5D or 6D tensor
If work with Recurrent, input is 6D tensor (`batch x time x nInputPlane x height x width x length`). output of
`forward(input)` is also expected to be a 6D tensor (`batch x time x outputPlane x height x width x length`).

If work with RecurrentDecoder, input is 5D tensor (`batch x nInputPlane x height x width x length`). output of
`forward(input)` is expected to be a 6D tensor (`batch x outputLen x outputPlane x height x width x length`).

Parameters:

* `inputSize` number of input planes in the image given into forward()
* `outputSize` number of output planes the convolution layer will produce
* `kernelI` convolutional filter size to convolve input
* `kernelC` convolutional filter size to convolve cell
* `stride` step of the convolution, default is 1
* `padding` step of the convolution, default is -1, behaves same with SAME padding in tensorflow
                 Default stride,padding value ensure last 3 dim of output shape is the same with input
* `wRegularizer` instance of [[Regularizer]] (eg. L1 or L2 regularization), applied to the input weights matrices.
* `uRegularizer` instance of [[Regularizer]] (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
* `bRegularizer` instance of [[Regularizer]], applied to the bias.
* `cRegularizer` instance of [[Regularizer]], applied to peephole.
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

val layer = TimeDistributed(Linear(3, 2))
val input = Tensor(2, 3, 3).rand()
layer.forward(input)
```
Input:
```
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.101178855	0.24703512	0.5021639
0.44016296	0.5694682	0.9227419
0.44305947	0.99880695	0.061260134

(2,.,.) =
0.7969414	0.20669454	0.27941006
0.22917499	0.21765763	0.22535545
0.389746	0.3487412	0.09982143

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x3]
```
Gives the output,
```
res0: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.38540328	-0.4002408
0.64361376	-0.33423418
0.4066636	-0.36263257

(2,.,.) =
0.023447769	-0.77664447
0.18752512	-0.53049827
0.13314348	-0.5799509

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x2]
```

**Python example:**
```python
from bigdl.nn.layer import TimeDistributed,Linear
import numpy as np

layer = TimeDistributed(Linear(3, 2))

input = np.random.random([2, 3, 3])
layer.forward(input)
```
Input:
```
array([[[ 0.3033118 ,  0.14485594,  0.58064829],
        [ 0.72854527,  0.5051743 ,  0.42110462],
        [ 0.78737995,  0.62032715,  0.20156085]],

       [[ 0.17852246,  0.72772084,  0.24014506],
        [ 0.01344367,  0.47754396,  0.65238232],
        [ 0.29103965,  0.50614159,  0.2816109 ]]])
```
Gives the output,
```
array([[[-0.10115834, -0.19001636],
        [-0.1446743 , -0.47479331],
        [-0.14148773, -0.61194205]],

       [[-0.28484675, -0.58061397],
        [-0.28640711, -0.29945394],
        [-0.18956462, -0.46879411]]], dtype=float32)
```
---

## MultiRNNCell ##

**Scala:**
```scala
// cells should be an array of Cell
val model = MultiRNNCell(cells = multiRNNCells)

```
**Python:**
```python
# cells should be a list of Cell
model = MultiRNNCell(cells = multiRNNCells)
```

A cell that stack multiple rnn cells(simpleRNN/LSTM/LSTMPeephole/GRU/ConvLSTMPeephole/ConvLSTMPeephole3D).
Only works with RecurrentDecoder. If you want to stack multiple cells with Recurrent. Use Sequential().add(Recurrent(cell)).add(Recurrent(cell))... instead

Parameters:

* `cells` list of RNNCell that will be composed in this order.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val hiddenSize = 2
val inputSize = 2
val batchSize = 2
val seqLength = 2
val input = Tensor(batchSize, inputSize, 3, 3).rand()
val gradOutput = Tensor(batchSize, seqLength, hiddenSize, 3, 3).rand()

val cells = Array(ConvLSTMPeephole(
  inputSize, hiddenSize, 3, 3, 1), ConvLSTMPeephole(
  inputSize, hiddenSize, 3, 3, 1)).asInstanceOf[Array[Cell[Float]]]
val model = RecurrentDecoder(seqLength).add(MultiRNNCell[Float](cells))

val output = model.forward(input)
val gradientInput = model.backward(input, gradOutput)

val states = model.getStates()
model.setStates(states)
-> print(output)
(1,1,1,.,.) =
0.035993136	0.04062611	0.038863156	
0.038338557	0.035591327	0.030849852	
0.03203216	0.026839556	0.033618193	

(1,1,2,.,.) =
-0.011673012	-0.013518209	-0.0079738535	
-0.013537201	-0.018129712	-0.013903147	
-0.015891023	-0.016045166	-0.015133085	

(1,2,1,.,.) =
0.051638972	0.06415851	0.0562743	
0.052649997	0.0433068	0.03683649	
0.0408955	0.0315791	0.043429054	

(1,2,2,.,.) =
-0.019818805	-0.024628056	-0.014551916	
-0.028422609	-0.036376823	-0.027259855	
-0.030024627	-0.033032943	-0.030440552	

(2,1,1,.,.) =
0.037235383	0.03971467	0.039468434	
0.032075796	0.031177454	0.029096292	
0.03708834	0.031535562	0.036211465	

(2,1,2,.,.) =
-0.010179557	-0.011387618	-0.008739926	
-0.013536877	-0.015962215	-0.017361978	
-0.014717996	-0.014296502	-0.016867846	

(2,2,1,.,.) =
0.053095814	0.05863748	0.05486801	
0.048524074	0.043160528	0.040398546	
0.04628137	0.04125476	0.043807983	

(2,2,2,.,.) =
-0.017849356	-0.019537563	-0.018888	
-0.025026768	-0.034455147	-0.02970969	
-0.026703741	-0.033036336	-0.027824042	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x2x3x3]
-> print(gradientInput)
(1,1,1,.,.) =
-0.021843424	-0.015910733	-0.013524098	
-0.019261343	-0.017457811	-0.013539563	
-0.016062422	-0.00383057	-0.0021248849	

(1,1,2,.,.) =
-0.0067594885	-0.012176989	-0.009976602	
-0.007914364	-0.012559764	-7.768459E-4	
-0.0026864496	-3.4671678E-4	-0.004467619	

(1,2,1,.,.) =
-0.011175868	-0.011886302	-0.0074315416	
-0.009660093	-0.009753445	-0.008733444	
-0.007047931	-0.0055002044	8.1458344E-4	

(1,2,2,.,.) =
-0.0016122719	-0.003776702	-0.006306042	
-0.0032693855	-0.005982614	-0.0010739439	
-0.0020354516	-9.59815E-4	-0.0010912241	

(2,1,1,.,.) =
-0.01399023	-0.01809205	-0.015330672	
-0.025769815	-0.00905557	-0.021059947	
4.068871E-4	-0.0060698274	-0.0048879837	

(2,1,2,.,.) =
-0.0013799625	-0.012721367	-0.008014497	
-0.014288196	-0.0185386	-0.017980032	
-0.0022621946	-0.015537363	-0.0024578157	

(2,2,1,.,.) =
-0.009561457	-0.007107652	-0.009356419	
-0.009839717	-0.0021937331	-0.011457165	
-0.0044140965	-0.0031195688	-0.0034824142	

(2,2,2,.,.) =
-3.2559165E-4	-0.0054697054	-0.0073612086	
-0.0014059425	-0.006272946	-0.0028436938	
0.0028391986	-0.005325649	-0.0028171889	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x3x3]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
import numpy as np
input_size = 2
output_size = 2
seq_length = 2
batch_size = 2
input = np.random.randn(batch_size, input_size, 3, 3)
grad_output = np.random.randn(batch_size, seq_length, output_size, 3, 3)
cells = []
cells.append(ConvLSTMPeephole(input_size, output_size, 3, 3, 1, with_peephole = False))
cells.append(ConvLSTMPeephole(input_size, output_size, 3, 3, 1, with_peephole = False))

model = RecurrentDecoder(seq_length).add(MultiRNNCell(cells))

output = model.forward(input)
gradient_input = model.backward(input, grad_output)

states = model.get_states()
model.set_states(states)
-> print output
[[[[[ 0.01858711  0.03114421  0.02070103]
    [ 0.01312863  0.00865137  0.02380039]
    [ 0.02127378  0.02221535  0.02805275]]

   [[ 0.05865936  0.06254016  0.07285608]
    [ 0.07795827  0.06420417  0.06744433]
    [ 0.07241444  0.06128554  0.0572256 ]]]


  [[[ 0.01813958  0.0388087   0.03606314]
    [ 0.00914392  0.01012017  0.03544089]
    [ 0.02192647  0.02542255  0.04978891]]

   [[ 0.06317041  0.07505058  0.10311646]
    [ 0.10012341  0.06632978  0.09895241]
    [ 0.10852461  0.08559311  0.07942865]]]]



 [[[[ 0.01352384  0.02394648  0.02436183]
    [ 0.00793007  0.01043395  0.03022798]
    [ 0.01539317  0.01955615  0.01543968]]

   [[ 0.05844339  0.05187995  0.05877664]
    [ 0.06405409  0.08493486  0.07711712]
    [ 0.0737301   0.05892281  0.05127344]]]


  [[[ 0.01918509  0.037876    0.04408969]
    [ 0.01470916  0.01985376  0.03152689]
    [ 0.02578159  0.04284319  0.0319238 ]]

   [[ 0.08844157  0.07580076  0.07929584]
    [ 0.09811849  0.08237181  0.09161879]
    [ 0.11196285  0.08747569  0.09312635]]]]]
    
-> print gradient_input
[[[[[-0.01967927  0.0118104   0.00034992]
    [-0.0132792  -0.0127134   0.01193821]
    [ 0.01297736  0.00550178  0.00874622]]

   [[-0.00718097  0.01717402  0.00893286]
    [-0.01143209  0.00079105  0.00920936]
    [ 0.01638926  0.02479215  0.01613754]]]


  [[[-0.02959971 -0.00214246 -0.00665301]
    [-0.02010076  0.00135842  0.01485039]
    [ 0.01877127  0.00205219 -0.01012903]]

   [[-0.01455194  0.00882864  0.00075077]
    [-0.0089175  -0.00774059  0.00534623]
    [ 0.00421638  0.01152828  0.00886414]]]]



 [[[[ 0.00945553  0.01345219 -0.01787379]
    [-0.02221245 -0.0047606   0.03430083]
    [ 0.01496986 -0.01156155  0.00733263]]

   [[ 0.02018309  0.00937438 -0.00253335]
    [-0.00616324  0.00972739  0.02758386]
    [ 0.01057806  0.01101648  0.00341856]]]


  [[[ 0.00486301 -0.00717946 -0.01368812]
    [-0.01296435  0.0466785  -0.0126987 ]
    [ 0.01161697 -0.01207331  0.01638841]]

   [[ 0.02077198 -0.00770913 -0.00807941]
    [-0.00096983  0.01721167  0.0265876 ]
    [ 0.00845431  0.01232574  0.0126167 ]]]]]

```

---
## Highway ##

**Scala:**
```scala
val layer = Highway(size, withBias = true,
                    activation = null,
                    wRegularizer = null,
                    bRegularizer = null)
```
**Python:**
```python
layer = Highway(size, with_bias=True,
                activation=None,
                wRegularizer=None,
                bRegularizer=None)
```

This layer is Densely connected highway network.
Highway layers are a natural extension of LSTMs to feedforward networks.

Parameters:

* `size` input size
* `with_bias` whether to include a bias
* `activation` activation function, by default no activation will be used.
  For Python, one can also pass the name of an existing activation as a string, eg. 'tanh', 'relu', 'sigmoid', 'hard_sigmoid', 'softmax' etc.
* `wRegularizer` instance of [[Regularizer]] (eg. L1 or L2 regularization), applied to the input weights matrices.
* `bRegularizer` instance of [[Regularizer]], applied to the bias.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val module = Highway(2, activation = Tanh())

val input = Tensor(3, 2).randn()
println(input)
val output = module.forward(input)
println(output)
```
Gives the output,
```
1.096164	0.08578972
0.2580359	1.629636
-0.7571692	0.28832582
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2]
0.65883696	0.108842306
-0.032798193	0.047720015
-0.5495165	-0.16949607
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input = np.random.rand(3, 2)
print "input is :",input

m = Highway(2, activation=Tanh())
out = m.forward(input)
print "output is :",out
```
Gives the output,
```
input is : [[ 0.65776902  0.63354682]
 [ 0.57766285  0.50117516]
 [ 0.15317826  0.60807496]]
creating: createHighway
output is : [[ 0.44779509 -0.10608637]
 [ 0.41307163 -0.14994906]
 [ 0.25687078  0.00718814]]
```

