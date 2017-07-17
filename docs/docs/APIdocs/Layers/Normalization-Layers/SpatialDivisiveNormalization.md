## SpatialDivisiveNormalization ##

**Scala:**
```scala
val layer = SpatialDivisiveNormalization()
```
**Python:**
```python
layer = SpatialDivisiveNormalization()
```

Applies a spatial division operation on a series of 2D inputs using kernel for
computing the weighted average in a neighborhood. The neighborhood is defined for
a local spatial region that is the size as kernel and across all features. For
an input image, since there is only one feature, the region is only spatial. For
an RGB image, the weighted average is taken over RGB channels and a spatial region.

If the kernel is 1D, then it will be used for constructing and separable 2D kernel.
The operations will be much more efficient in this case.

The kernel is generally chosen as a gaussian when it is believed that the correlation
of two pixel locations decrease with increasing distance. On the feature dimension,
a uniform average is used since the weighting across features is not known.

**Scala example:**
```scala

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val layer = SpatialDivisiveNormalization()
val input = Tensor(1, 5, 5).rand
val gradOutput = Tensor(1, 5, 5).rand

val output = layer.forward(input)
val gradInput = layer.backward(input, gradOutput)

> println(input)
res19: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.4022106       0.6872489       0.9712838       0.7769542       0.771034
0.97930336      0.61022973      0.65092266      0.9507807       0.3158211
0.12607759      0.320569        0.9267993       0.47579524      0.63989824
0.713135        0.30836385      0.009723447     0.67723924      0.24405171
0.51036286      0.115807846     0.123513035     0.28398398      0.271164

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5x5]

> println(output)
res20: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.37849638      0.6467289       0.91401714      0.73114514      0.725574
0.9215639       0.57425076      0.6125444       0.89472294      0.29720038
0.11864409      0.30166835      0.8721555       0.4477425       0.60217
0.67108876      0.2901828       0.009150156     0.6373094       0.2296625
0.480272        0.10897984      0.11623074      0.26724035      0.25517625

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5x5]

> println(gradInput)
res21: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.09343022     -0.25612304     0.25756648      -0.66132677     -0.44575396
0.052990615     0.7899354       0.27205157      0.028260134     0.23150417
-0.115425855    0.21133065      0.53093016      -0.36421964     -0.102551565
0.7222408       0.46287358      0.0010696054    0.26336592      -0.050598443
0.03733714      0.2775169       -0.21430963     0.3175013       0.6600435

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x5x5]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

layer = SpatialDivisiveNormalization()
input = np.random.uniform(0, 1, (1, 5, 5)).astype("float32")
gradOutput = np.random.uniform(0, 1, (1, 5, 5)).astype("float32")

output = layer.forward(input)
gradInput = layer.backward(input, gradOutput)

> output
[array([[[ 0.30657911,  0.75221181,  0.2318386 ,  0.84053135,  0.24818985],
         [ 0.32852787,  0.43504578,  0.0219258 ,  0.47856906,  0.31112722],
         [ 0.12381417,  0.61807972,  0.90043157,  0.57342309,  0.65450585],
         [ 0.00401461,  0.33700454,  0.79859954,  0.64382601,  0.51768768],
         [ 0.38087726,  0.8963666 ,  0.7982524 ,  0.78525543,  0.09658573]]], dtype=float32)]
> gradInput
[array([[[ 0.08059166, -0.4616771 ,  0.11626807,  0.30253756,  0.7333734 ],
         [ 0.2633073 , -0.01641282,  0.40653706,  0.07766753, -0.0237394 ],
         [ 0.10733987,  0.23385212, -0.3291783 , -0.12808481,  0.4035565 ],
         [ 0.56126803,  0.49945205, -0.40531909, -0.18559581,  0.27156472],
         [ 0.28016835,  0.03791744, -0.17803842, -0.27817759,  0.42473239]]], dtype=float32)]
```
