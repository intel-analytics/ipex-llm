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
Ref. A.: http://arxiv.org/pdf/1303.5778v1 (blueprint for this module)
B. http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf
C. http://arxiv.org/pdf/1503.04069v1.pdf
D. https://github.com/wojzaremba/lstm

- param inputSize the size of each input vector
- param hiddenSize Hidden unit size in the LSTM
- param  p is used for [[Dropout]] probability. For more details about
           RNN dropouts, please refer to
           [RnnDrop: A Novel Dropout for RNNs in ASR]
           (http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)
           [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks]
           (https://arxiv.org/pdf/1512.05287.pdf)
- param wRegularizer: instance of [[Regularizer]]
                   (eg. L1 or L2 regularization), applied to the input weights matrices.
- param uRegularizer: instance [[Regularizer]]
          (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
- param bRegularizer: instance of [[Regularizer]]
          applied to the bias.

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
