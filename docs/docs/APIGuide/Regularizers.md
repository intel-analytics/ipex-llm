## L1 Regularizer ##

**Scala:**
```scala
val l1Regularizer = L1Regularizer(rate)
```
**Python:**
```python
regularizerl1 = L1Regularizer(rate)
```

L1 regularizer is used to add penalty to the gradWeight to avoid overfitting.

In our code implementation, gradWeight = gradWeight + alpha * abs(weight)

For more details, please refer to [wiki](https://en.wikipedia.org/wiki/Regularization_(mathematics)).

**Scala example:**
```scala

import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn._

RNG.setSeed(100)

val input = Tensor(3, 5).rand
val gradOutput = Tensor(3, 5).rand
val linear = Linear(5, 5, wRegularizer = L1Regularizer(0.2), bRegularizer = L1Regularizer(0.2))

val output = linear.forward(input)
val gradInput = linear.backward(input, gradOutput)

scala> input
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.54340494      0.67115563      0.2783694       0.4120464       0.4245176
0.52638245      0.84477615      0.14860484      0.004718862     0.15671109
0.12156912      0.18646719      0.67074907      0.21010774      0.82585275
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x5]

scala> gradOutput
gradOutput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.4527399       0.13670659      0.87014264      0.5750933       0.063681036
0.89132196      0.62431186      0.20920213      0.52334774      0.18532822
0.5622963       0.10837689      0.0058171963    0.21969749      0.3074232
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x5]

scala> linear.gradWeight
res2: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.9835552       1.3616763       0.83564335      0.108898684     0.59625006
0.21608911      0.8393639       0.0035243928    -0.11795368     0.4453743
0.38366735      0.9618148       0.47721142      0.5607486       0.6069793
0.81469804      0.6690552       0.18522228      0.08559488      0.7075894
-0.030468717    0.056625083     0.051471338     0.2917061       0.109963015
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5x5]

```

**Python example:**
```python

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

input = np.random.uniform(0, 1, (3, 5)).astype("float32")
gradOutput = np.random.uniform(0, 1, (3, 5)).astype("float32")
linear = Linear(5, 5, wRegularizer = L1Regularizer(0.2), bRegularizer = L1Regularizer(0.2))
output = linear.forward(input)
gradInput = linear.backward(input, gradOutput)

> linear.parameters()
{u'Linear@596d857b': {u'bias': array([ 0.3185505 , -0.02004393,  0.34620118, -0.09206461,  0.40776938], dtype=float32),
  u'gradBias': array([ 2.14087653,  1.82181644,  1.90674937,  1.37307787,  0.81534696], dtype=float32),
  u'gradWeight': array([[ 0.34909648,  0.85083449,  1.44904375,  0.90150446,  0.57136625],
         [ 0.3745544 ,  0.42218602,  1.53656614,  1.1836741 ,  1.00702667],
         [ 0.30529332,  0.26813674,  0.85559171,  0.61224306,  0.34721529],
         [ 0.22859855,  0.8535381 ,  1.19809723,  1.37248564,  0.50041491],
         [ 0.36197871,  0.03069445,  0.64837945,  0.12765063,  0.12872688]], dtype=float32),
  u'weight': array([[-0.12423037,  0.35694697,  0.39038274, -0.34970999, -0.08283543],
         [-0.4186025 , -0.33235055,  0.34948507,  0.39953214,  0.16294235],
         [-0.25171402, -0.28955361, -0.32243955, -0.19771226, -0.29320192],
         [-0.39263198,  0.37766701,  0.14673658,  0.24882999, -0.0779015 ],
         [ 0.0323218 , -0.31266898,  0.31543773, -0.0898933 , -0.33485892]], dtype=float32)}}
```




## L2 Regularizer ##

**Scala:**
```scala
val l2Regularizer = L2Regularizer(rate)
```
**Python:**
```python
regularizerl2 = L2Regularizer(rate)
```

L2 regularizer is used to add penalty to the gradWeight to avoid overfitting.

In our code implementation, gradWeight = gradWeight + alpha * weight * weight

For more details, please refer to [wiki](https://en.wikipedia.org/wiki/Regularization_(mathematics)).

**Scala example:**
```scala

import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn._

RNG.setSeed(100)

val input = Tensor(3, 5).rand
val gradOutput = Tensor(3, 5).rand
val linear = Linear(5, 5, wRegularizer = L2Regularizer(0.2), bRegularizer = L2Regularizer(0.2))

val output = linear.forward(input)
val gradInput = linear.backward(input, gradOutput)

scala> input
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.54340494      0.67115563      0.2783694       0.4120464       0.4245176
0.52638245      0.84477615      0.14860484      0.004718862     0.15671109
0.12156912      0.18646719      0.67074907      0.21010774      0.82585275
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x5]

scala> gradOutput
gradOutput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.4527399       0.13670659      0.87014264      0.5750933       0.063681036
0.89132196      0.62431186      0.20920213      0.52334774      0.18532822
0.5622963       0.10837689      0.0058171963    0.21969749      0.3074232
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x5]

scala> linear.gradWeight
res0: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0329735       0.047239657     0.8979603       0.53614384      1.2781229
0.5621818       0.29772854      0.69706535      0.30559152      0.8352279
1.3044653       0.43065858      0.9896795       0.7435816       1.6003494
0.94218314      0.6793372       0.97101355      0.62892824      1.3458569
0.73134506      0.5975239       0.9109101       0.59374434      1.1656629
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5x5]

```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

input = np.random.uniform(0, 1, (3, 5)).astype("float32")
gradOutput = np.random.uniform(0, 1, (3, 5)).astype("float32")
linear = Linear(5, 5, wRegularizer = L2Regularizer(0.2), bRegularizer = L2Regularizer(0.2))
output = linear.forward(input)
gradInput = linear.backward(input, gradOutput)

> linear.parameters()
{u'Linear@787aab5e': {u'bias': array([-0.43960261, -0.12444571,  0.22857292, -0.43216187,  0.27770036], dtype=float32),
  u'gradBias': array([ 0.51726723,  1.32883406,  0.57567948,  1.7791357 ,  1.2887038 ], dtype=float32),
  u'gradWeight': array([[ 0.45477036,  0.22262168,  0.21923628,  0.26152173,  0.19836383],
         [ 1.12261093,  0.72921795,  0.08405925,  0.78192139,  0.48798928],
         [ 0.34581488,  0.21195598,  0.26357424,  0.18987852,  0.2465664 ],
         [ 1.18659711,  1.11271608,  0.72589797,  1.19098675,  0.33769298],
         [ 0.82314551,  0.71177536,  0.4428404 ,  0.764337  ,  0.3500182 ]], dtype=float32),
  u'weight': array([[ 0.03727285, -0.39697152,  0.42733836, -0.34291714, -0.13833708],
         [ 0.09232076, -0.09720675, -0.33625153,  0.06477787, -0.34739712],
         [ 0.17145753,  0.10128133,  0.16679128, -0.33541158,  0.40437087],
         [-0.03005157, -0.36412898,  0.0629965 ,  0.13443278, -0.38414535],
         [-0.16630849,  0.06934392,  0.40328237,  0.22299488, -0.1178569 ]], dtype=float32)}}
```

## L1L2 Regularizer ##

**Scala:**
```scala
val l1l2Regularizer = L1L2Regularizer(l1rate, l2rate)
```
**Python:**
```python
regularizerl1l2 = L1L2Regularizer(l1rate, l2rate)
```

L1L2 regularizer is used to add penalty to the gradWeight to avoid overfitting.

In our code implementation, we will apply L1regularizer and L2regularizer sequentially.

For more details, please refer to [wiki](https://en.wikipedia.org/wiki/Regularization_(mathematics)).

**Scala example:**
```scala

import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn._

RNG.setSeed(100)

val input = Tensor(3, 5).rand
val gradOutput = Tensor(3, 5).rand
val linear = Linear(5, 5, wRegularizer = L1L2Regularizer(0.2, 0.2), bRegularizer = L1L2Regularizer(0.2, 0.2))

val output = linear.forward(input)
val gradInput = linear.backward(input, gradOutput)

scala> input
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.54340494      0.67115563      0.2783694       0.4120464       0.4245176
0.52638245      0.84477615      0.14860484      0.004718862     0.15671109
0.12156912      0.18646719      0.67074907      0.21010774      0.82585275
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x5]

scala> gradOutput
gradOutput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.4527399       0.13670659      0.87014264      0.5750933       0.063681036
0.89132196      0.62431186      0.20920213      0.52334774      0.18532822
0.5622963       0.10837689      0.0058171963    0.21969749      0.3074232
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x5]

scala> linear.gradWeight
res1: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.069174        1.4422078       0.8913989       0.042112567     0.53756505
0.14077617      0.8959319       -0.030221784    -0.1583686      0.4690558
0.37145022      0.99747723      0.5559263       0.58614403      0.66380215
0.88983417      0.639738        0.14924419      0.027530536     0.71988696
-0.053217214    -8.643427E-4    -0.036953792    0.29753304      0.06567569
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5x5]
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

input = np.random.uniform(0, 1, (3, 5)).astype("float32")
gradOutput = np.random.uniform(0, 1, (3, 5)).astype("float32")
linear = Linear(5, 5, wRegularizer = L1L2Regularizer(0.2, 0.2), bRegularizer = L1L2Regularizer(0.2, 0.2))
output = linear.forward(input)
gradInput = linear.backward(input, gradOutput)

> linear.parameters()
{u'Linear@1356aa91': {u'bias': array([-0.05799473, -0.0548001 ,  0.00408955, -0.22004321, -0.07143869], dtype=float32),
  u'gradBias': array([ 0.89119786,  1.09953558,  1.03394508,  1.19511735,  2.02241182], dtype=float32),
  u'gradWeight': array([[ 0.89061081,  0.58810186, -0.10087357,  0.19108151,  0.60029608],
         [ 0.95275503,  0.2333075 ,  0.46897018,  0.74429053,  1.16038764],
         [ 0.22894514,  0.60031962,  0.3836292 ,  0.15895618,  0.83136207],
         [ 0.49079862,  0.80913013,  0.55491877,  0.69608945,  0.80458677],
         [ 0.98890561,  0.49226439,  0.14861123,  1.37666655,  1.47615671]], dtype=float32),
  u'weight': array([[ 0.44654208,  0.16320795, -0.36029238, -0.25365737, -0.41974261],
         [ 0.18809238, -0.28065765,  0.27677274, -0.29904234,  0.41338971],
         [-0.03731538,  0.22493915,  0.10021331, -0.19495697,  0.25470355],
         [-0.30836752,  0.12083009,  0.3773002 ,  0.24059358, -0.40325543],
         [-0.13601269, -0.39310011, -0.05292636,  0.20001481, -0.08444868]], dtype=float32)}}
```
