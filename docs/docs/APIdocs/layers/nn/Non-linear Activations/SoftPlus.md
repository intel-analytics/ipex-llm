## SoftPlus ##

**Scala:**
```scala
val model = SoftPlus(beta = 1.0)
```
**Python:**
```python
model = SoftPlus(beta = 1.0)
```

Apply the SoftPlus function to an n-dimensional input tensor.
SoftPlus function: 
```
f_i(x) = 1/beta * log(1 + exp(beta * x_i))
```
- param beta Controls sharpness of transfer function

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

val model = SoftPlus()
val input = Tensor(2, 3, 4).rand()
val output = model.forward(input)

scala> println(input)
(1,.,.) =
0.9812126	0.7044107	0.0657767	0.9173636	
0.20853543	0.76482195	0.60774535	0.47837523	
0.62954164	0.56440496	0.28893307	0.40742245	

(2,.,.) =
0.18701692	0.7700966	0.98496467	0.8958407	
0.037015386	0.34626052	0.36459026	0.8460807	
0.051016055	0.6742781	0.14469075	0.07565566	

scala> println(output)
(1,.,.) =
1.2995617	1.1061354	0.7265762	1.2535294	
0.80284095	1.1469617	1.0424956	0.9606715	
1.0566612	1.0146512	0.8480129	0.91746557	

(2,.,.) =
0.7910212	1.1505641	1.3022922	1.2381986	
0.71182615	0.88119024	0.8919668	1.203121	
0.7189805	1.0860726	0.7681072	0.7316903	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

model = SoftPlus()
input = np.random.randn(2, 3, 4)
output = model.forward(input)

>>> print(input)
[[[ 0.82634972 -0.09853824  0.97570235  1.84464617]
  [ 0.38466503  0.08963732  1.29438774  1.25204527]
  [-0.01910449 -0.19560752 -0.81769143 -1.06365733]]

 [[-0.56284365 -0.28473239 -0.58206869 -1.97350909]
  [-0.28303919 -0.59735361  0.73282102  0.0176838 ]
  [ 0.63439133  1.84904987 -1.24073643  2.13275833]]]
>>> print(output)
[[[ 1.18935537  0.6450913   1.2955569   1.99141073]
  [ 0.90386271  0.73896986  1.53660071  1.50351918]
  [ 0.68364054  0.60011864  0.36564925  0.29653603]]

 [[ 0.45081255  0.56088102  0.44387865  0.1301229 ]
  [ 0.56160825  0.43842646  1.12523568  0.70202816]
  [ 1.0598278   1.99521446  0.2539995   2.24475574]]]
```
