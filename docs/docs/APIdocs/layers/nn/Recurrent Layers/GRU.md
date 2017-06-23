## GRU ##

**Scala:**
```scala
val gru = GRU(inputSize, ouputSize)
```
**Python:**
```python
gru = GRU(inputSize, outputSize)
```

Gated Recurrent Units architecture. The first input in sequence uses zero value for cell and hidden state.

Ref.
 1. http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
 2. https://github.com/Element-Research/rnn/blob/master/GRU.lua
 
 
**Parameters:**
* **inputSize** - the size of each input vector
* **outputSize** - hidden unit size in GRU
* **p** - is used for [[Dropout]] probability. For more details about
          RNN dropouts, please refer to
           [RnnDrop: A Novel Dropout for RNNs in ASR](http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)
            and [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf). Default: 0.0
* **wRegularizer** - instance of `Regularizer`(eg. L1 or L2 regularization), applied to the input weights matrices. Default: null
* **uRegularizer** - instance of `Regularizer`(eg. L1 or L2 regularization), applied to the recurrent weights matrices. Default: null
* **bRegularizer** - instance of `Regularizer`(eg. L1 or L2 regularization), applied to the bias. Default: null

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
val hiddenSize = 2
val inputSize = 2
val outputSize = 2
val seqLength = 2
val input = Tensor[Float](T(
  T(1.0f, 2.0f),
  T(2.0f, 3.0f)
)).resize(Array(1, seqLength, inputSize))
val target = Tensor[Float](T(
  T(2.0f, 3.0f),
  T(4.0f, 5.0f)
)).resize(Array(1, seqLength, inputSize))
val rec = Recurrent[Float]()

val model = Sequential[Float]()
    .add(rec.add(GRU[Float](inputSize, hiddenSize)))
    .add(TimeDistributed[Float](Linear[Float](hiddenSize, outputSize)))
val output = model.forward(input)
val gradient = model.backward(input, target)

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
target = np.array([[
  [2.0, 3.0],
  [4.0, 5.0]
]])
rec = Recurrent()

model = Sequential() \
    .add(rec.add(GRU(input_size, hidden_size))) \
    .add(TimeDistributed(Linear(hidden_size, output_size)))
output = model.forward(input)
gradient = model.backward(input, target)
-> print output
# There's random factor. An output could be
[[[ 0.27857888  0.20263115]
  [ 0.29470384  0.22594413]]]
-> print gradient
[[[-0.32956457  0.27405274]
  [-0.32718879  0.32963118]]]
```
