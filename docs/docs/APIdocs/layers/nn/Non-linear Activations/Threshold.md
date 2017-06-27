## Threshold ##

**Scala:**
```scala
val module = Threshold(threshold, value, ip)
```
**Python:**
```python
module = Threshold(threshold, value, ip)
```

Thresholds each element of the input Tensor.
Threshold is defined as:

```
     ⎧ x        if x >= threshold
 y = ⎨ 
     ⎩ value    if x <  threshold
```

- threshold: The value to threshold at
- value: The value to replace with
- ip: can optionally do the operation in-place

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val module = Threshold(1, 0.8)
val input = Tensor(2, 2, 2).randn()
val output = module.forward(input)

> input
(1,.,.) =
2.0502799	-0.37522468
-1.2704345	-0.22533786

(2,.,.) =
1.1959263	1.6670992
-0.24333914	1.4424673

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2]

> output
(1,.,.) =
(1,.,.) =
2.0502799	0.8
0.8	0.8

(2,.,.) =
1.1959263	1.6670992
0.8	1.4424673

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Threshold(1.0, 0.8)
input = np.random.randn(2, 2, 2)
output = module.forward(input)

> input
[[[-0.43226865 -1.09160093]
  [-0.20280088  0.68196767]]

 [[ 2.32017942  1.00003307]
  [-0.46618767  0.57057167]]]
  
> output
[array([[[ 0.80000001,  0.80000001],
        [ 0.80000001,  0.80000001]],

       [[ 2.32017946,  1.00003302],
        [ 0.80000001,  0.80000001]]], dtype=float32)]
```
