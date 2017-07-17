## Mul ##

**Scala:**
```scala
val module = Mul()
```
**Python:**
```python
module = Mul()
```

Multiply a singla scalar factor to the incoming data

```
                 +----Mul----+
 input -----+---> input * weight -----+----> output
```

**Scala example:**
```scala

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val mul = Mul()

> print(mul.forward(Tensor(1, 5).rand()))
-0.03212923     -0.019040342    -9.136753E-4    -0.014459004    -0.04096878
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

mul = Mul()
input = np.random.uniform(0, 1, (1, 5)).astype("float32")

> mul.forward(input)
[array([[ 0.72429317,  0.7377845 ,  0.09136307,  0.40439236,  0.29011244]], dtype=float32)]

```
