## Sequential ##

**Scala:**
```scala
val module = Sequential()
```
**Python:**
```python
seq = Sequential()
```

Sequential provides a means to plug layers together
in a feed-forward fully connected manner.

**Scala example:**

```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

import com.intel.analytics.bigdl.nn.{Sequential, Linear}

val module = Sequential()
module.add(Linear(10, 25))
module.add(Linear(25, 10))

val input = Tensor(10).range(1, 10, 1)
val gradOutput = Tensor(10).range(1, 10, 1)

val output = module.forward(input).toTensor
val gradInput = module.backward(input, gradOutput).toTensor

println(output)
println(gradInput)
```

The output is,

```
-2.3750305
2.4512818
1.6998017
-0.47432393
4.3048754
-0.044168986
-1.1643536
0.60341483
2.0216258
2.1190155
[com.intel.analytics.bigdl.tensor.DenseTensor of size 10]
```

The gradInput is,

```
2.593382
-1.4137214
-1.8271983
1.229643
0.51384985
1.509845
2.9537349
1.088281
0.2618509
1.4840821
[com.intel.analytics.bigdl.tensor.DenseTensor of size 10]
```

**Python example:**

```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

seq = Sequential()
seq.add(Linear(10, 25))
seq.add(Linear(25, 10))

input = np.arange(1, 11, 1).astype("float32")
input = input.reshape(1, 10)

output = seq.forward(input)
print output

gradOutput = np.arange(1, 11, 1).astype("float32")
gradOutput = gradOutput.reshape(1, 10)

gradInput = seq.backward(input, gradOutput)
print gradInput
```

The output is,

```
[array([[ 1.08462083, -2.03257799, -0.5400058 ,  0.27452484,  1.85562158,
         1.64338267,  2.45694995,  1.70170391, -2.12998056, -1.28924525]], dtype=float32)]
```

The gradInput is,

```

[array([[ 1.72007763,  1.64403224,  2.52977395, -1.00021958,  0.1134415 ,
         2.06711197,  2.29631734, -3.39587498,  1.01093054, -0.54482007]], dtype=float32)]
```
