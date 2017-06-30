## SmoothL1Criterion ##

**Scala:**
```scala
val slc = SmoothL1Criterion(sizeAverage=true)
```
**Python:**
```python
slc = SmoothL1Criterion(size_average=True)
```
Creates a criterion that can be thought of as a smooth version of the AbsCriterion.
It uses a squared term if the absolute element-wise error falls below 1.
It is less sensitive to outliers than the MSECriterion and in some
cases prevents exploding gradients (e.g. see "Fast R-CNN" paper by Ross Girshick).

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.{Tensor, Storage}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn.SmoothL1Criterion

val slc = SmoothL1Criterion()

val inputArr = Array(
  0.17503996845335,
  0.83220188552514,
  0.48450597329065,
  0.64701424003579,
  0.62694586534053,
  0.34398410236463,
  0.55356747563928,
  0.20383032318205
)
val targetArr = Array(
  0.69956525065936,
  0.86074831243604,
  0.54923197557218,
  0.57388074393384,
  0.63334444304928,
  0.99680578662083,
  0.49997645849362,
  0.23869121982716
)

val input = Tensor(Storage(inputArr.map(x => x.toFloat))).reshape(Array(2, 2, 2))
val target = Tensor(Storage(targetArr.map(x => x.toFloat))).reshape(Array(2, 2, 2))

val output = slc.forward(input, target)
val gradInput = slc.backward(input, target)
```

The output is,

```
output: Float = 0.0447365
```

The gradInput is,

```
gradInput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.06556566     -0.003568299
-0.008090746    0.009141691

(2,.,.) =
-7.998273E-4    -0.08160271
0.0066988766    -0.0043576136
```


**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

slc = SmoothL1Criterion()

input = np.array([
    0.17503996845335,
    0.83220188552514,
    0.48450597329065,
    0.64701424003579,
    0.62694586534053,
    0.34398410236463,
    0.55356747563928,
    0.20383032318205
])
input.reshape(2, 2, 2)

target = np.array([
    0.69956525065936,
    0.86074831243604,
    0.54923197557218,
    0.57388074393384,
    0.63334444304928,
    0.99680578662083,
    0.49997645849362,
    0.23869121982716
])

target.reshape(2, 2, 2)

output = slc.forward(input, target)
gradInput = slc.backward(input, target)

print output
print gradInput
```
