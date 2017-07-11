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
