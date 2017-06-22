## LSTMPeephole ##

**Scala:**
```scala
val model = LSTMPeephole[Float](inputSize, hiddenSize)
```
**Python:**
```python
model = LSTMPeephole(inputSize, hiddenSize)
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
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

val hiddenSize = 4
val inputSize = 6
val outputSize = 5
val seqLength = 5
val batchSize = 1
               
val input = Tensor[Float](Array(batchSize, seqLength, inputSize))
for (b <- 1 to batchSize) {
  for (i <- 1 to seqLength) {
    val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0) * inputSize).toInt
    input.setValue(b, i, rdmInput, 1.0f)
  }
}

val rec = Recurrent[Float](hiddenSize)
val model = Sequential[Float]().add(rec.add(LSTMPeephole[Float](inputSize, hiddenSize))).add(TimeDistributed[Float](Linear[Float](hiddenSize, outputSize)))
val output = model.forward(input).toTensor
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
(1,.,.) =
0.35383725	-0.22476536	-0.46047324	-0.26038578	-0.21095484	
0.3409024	-0.22834192	-0.41133574	-0.27646995	-0.23721263	
0.39881697	-0.18804908	-0.48271912	-0.29778507	-0.14873621	
0.43038777	-0.16956224	-0.46273726	-0.30802295	-0.12813234	
0.32592735	-0.24277578	-0.42178982	-0.27876818	-0.23236775	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5x5]

```
**Python example:**
```python
hiddenSize = 4
inputSize = 6
outputSize = 5
seqLength = 5
batchSize = 1
               
input = np.random.randn(batchSize, seqLength, inputSize)
rec = Recurrent(hiddenSize)
model = Sequential().add(rec.add(LSTMPeephole(inputSize, hiddenSize))).add(TimeDistributed(Linear(hiddenSize, outputSize)))
output = model.forward(input)
```
output is
```
array([[[ 0.38146877,  0.08686808, -0.16959271, -0.13243586, -0.02830471],
        [ 0.26316535,  0.06359337, -0.22447851, -0.06319767, -0.13872764],
        [ 0.42890453,  0.17883307, -0.20073381, -0.06245731, -0.04297322],
        [ 0.29129934,  0.17688879, -0.2768988 ,  0.11385346, -0.23123962],
        [ 0.40386844,  0.21273491, -0.2435573 , -0.05527414, -0.04689732]]], dtype=float32)
```
