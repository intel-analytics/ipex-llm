## Sqrt ##

Apply an element-wise sqrt operation.

**Scala:**

```scala
val sqrt = new Sqrt
```

**Python:**
```python
sqrt = Sqrt()
```

Apply an element-wise sqrt operation.


**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Sqrt
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val input = Tensor(3, 5).range(1, 15, 1)
val sqrt = new Sqrt
val output = sqrt.forward(input)
println(output)

val gradOutput = Tensor(3, 5).range(2, 16, 1)
val gradInput = sqrt.backward(input, gradOutput)
println(gradOutput
```

The output will be,

```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0     1.4142135       1.7320508       2.0     2.236068
2.4494898       2.6457512       2.828427        3.0     3.1622777
3.3166249       3.4641016       3.6055512       3.7416575       3.8729835
```

The gradInput will be,

```
gradInput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0     1.0606601       1.1547005       1.25    1.3416407
1.428869        1.5118579       1.5909902       1.6666667       1.7392527
1.8090681       1.8763883       1.9414507       2.0044594       2.065591
```

**Python example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

sqrt = Sqrt()

input = np.arange(1, 16, 1).astype("float32")
input = input.reshape(3, 5)

output = sqrt.forward(input)
print output

gradOutput = np.arange(2, 17, 1).astype("float32")
gradOutput = gradOutput.reshape(3, 5)

gradInput = sqrt.backward(input, gradOutput)
print gradInput
```

The output will be:

```
[array([[ 1.        ,  1.41421354,  1.73205078,  2.        ,  2.23606801],
       [ 2.44948983,  2.64575124,  2.82842708,  3.        ,  3.1622777 ],
       [ 3.31662488,  3.46410155,  3.60555124,  3.7416575 ,  3.87298346]], dtype=float32)]
```

The gradInput will be:

```
[array([[ 1.        ,  1.06066012,  1.15470052,  1.25      ,  1.34164071],
       [ 1.42886901,  1.51185787,  1.59099019,  1.66666675,  1.73925269],
       [ 1.80906808,  1.87638831,  1.94145072,  2.00445938,  2.0655911 ]], dtype=float32)]
```