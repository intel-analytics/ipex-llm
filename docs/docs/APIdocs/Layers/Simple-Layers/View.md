## View ##

**Scala:**

```scala
val view = View(2, 8)
```

or

```
val view = View(Array(2, 8))
```

**Python:**
```python
view = View([2, 8])
```

This module creates a new view of the input tensor using the sizes passed to the constructor.
The method setNumInputDims() allows to specify the expected number of dimensions of the inputs
of the modules. This makes it possible to use minibatch inputs
when using a size -1 for one of the dimensions.

**Scala example:**

```scala
import com.intel.analytics.bigdl.nn.View
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val view = View(2, 8)

val input = Tensor(4, 4).randn()
val gradOutput = Tensor(2, 8).randn()

val output = view.forward(input)
val gradInput = view.backward(input, gradOutput)
```

The output is,

```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.43037438     1.2982363       -1.4723133      -0.2602826      0.7178128       -1.8763185      0.88629466      0.8346704
0.20963766      -0.9349786      1.0376515       1.3153045       1.5450214       1.084113        -0.29929757     -0.18356979
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x8]
```

The gradInput is,

```
gradInput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.7360089       0.9133299       0.40443268      -0.94965595
0.80520976      -0.09671917     -0.5498001      -0.098691925
-2.3119886      -0.8455147      0.75891125      1.2985301
0.5023749       1.4983269       0.42038065      -1.7002305
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

view = View([2, 8])

input = np.random.uniform(0, 1, [4, 4]).astype("float32")
gradOutput = np.random.uniform(0, 1, [2, 8]).astype("float32")

output = view.forward(input)
gradInput = view.backward(input, gradOutput)

print output
print gradInput
```
