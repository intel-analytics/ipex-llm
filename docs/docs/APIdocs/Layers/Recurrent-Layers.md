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

> input
(1,.,.) =
0.0	1.0	0.0	0.0	0.0
0.0	1.0	0.0	0.0	0.0
0.0	0.0	1.0	0.0	0.0
0.0	1.0	0.0	0.0	0.0
0.0	0.0	1.0	0.0	0.0

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x5x5]

> output
(1,.,.) =
-0.44992247	-0.50529593	-0.033753205	-0.29562786
-0.19734861	-0.5647412	0.07520321	-0.35515767
-0.6771096	-0.4985356	-0.5806829	-0.47552463
-0.06949129	-0.53153497	0.11510986	-0.34098053
-0.71635246	-0.5226476	-0.5929389	-0.46533492

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

> output
[array([[[ 0.7526533 ,  0.29162994, -0.28749418, -0.11243925],
         [ 0.33291328, -0.07243762, -0.38017112,  0.53216213],
         [ 0.83854133,  0.07213539, -0.34503224,  0.33690596],
         [ 0.44095358,  0.27467242, -0.05471399,  0.46601957],
         [ 0.451913  , -0.33519334, -0.61357468,  0.56650752]]], dtype=float32)]
```

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
 * @param merge concat or add the output tensor of the two RNNs. Default is add

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
  with_peephole = True)
```

Convolution Long Short Term Memory architecture with peephole.
Ref. A.: https://arxiv.org/abs/1506.04214 (blueprint for this module)
B. https://github.com/viorik/ConvLSTM

- param inputSize: number of input planes in the image given into forward()
- param outputSize: number of output planes the convolution layer will produce
- param kernelI: convolutional filter size to convolve input
- param kernelC: convolutional filter size to convolve cell
- param stride: step of the convolution
- param wRegularizer: instance of [[Regularizer]]
                   (eg. L1 or L2 regularization), applied to the input weights matrices.
- param uRegularizer: instance [[Regularizer]]
          (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
- param bRegularizer: instance of [[Regularizer]]
          applied to the bias.
- param withPeephole: whether use last cell status control a gate

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

