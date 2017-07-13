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
