## Bottle ##

**Scala:**
```scala
val model = Bottle(module, nInputDim, nOutputDim)
```
**Python:**
```python
model = Bottle(module, nInputDim, nOutputDim)
```

Bottle allows varying dimensionality input to be forwarded through any module that accepts input of nInputDim dimensions, and generates output of nOutputDim dimensions.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val model = Bottle(Linear(3, 2), 2, 2)
val input = Tensor(2, 3, 3).rand()

scala> print(input)
(1,.,.) =
0.7843752	0.17286697	0.20767091	
0.8594811	0.9100018	0.8448141	
0.7683892	0.36661968	0.76637685	

(2,.,.) =
0.7163263	0.083962396	0.81222403	
0.7947034	0.09976136	0.114404656	
0.14890474	0.43289232	0.1489096	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3x3]	

val output = model.forward(input)

scala> print(output)
(1,.,.) =
-0.31146684	0.40719786	
-0.51778656	0.58715886	
-0.51676923	0.4027511	

(2,.,.) =
-0.5498678	0.29658738	
-0.280177	0.39901164	
-0.2387946	0.24809375	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x2]
```

**Python example:**
```python
model = Bottle(Linear(3, 2), 2, 2)

input = np.random.randn(2, 3, 3)
output = model.forward(input)

>>> print(input)
[[[ 0.42370589 -1.7938942   0.56666373]
  [-1.78501381  0.55676471 -0.50150367]
  [-1.59262182  0.82079469  1.1873599 ]]

 [[ 0.95799792 -0.71447244  1.05344083]
  [-0.07838376 -0.88780484 -1.80491177]
  [ 0.99996222  1.39876002 -0.16326094]]]
>>> print(output)
[[[ 0.26298434  0.74947536]
  [-1.24375117 -0.33148435]
  [-1.35218966  0.17042145]]

 [[ 0.08041853  0.91245329]
  [-0.08317742 -0.13909879]
  [-0.52287608  0.3667658 ]]]
```
