## Clamp ##

**Scala:**
```scala
val model = Clamp(min, max)
```
**Python:**
```python
model = Clamp(min, max)
```

A kind of hard tanh activition function with integer min and max
- param min min value
- param max max value

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

val model = Clamp(-10, 10)
val input = Tensor(2, 2, 2).rand()
val output = model.forward(input)

scala> print(input)
(1,.,.) =
0.95979714	0.27654588	
0.35592428	0.49355772	

(2,.,.) =
0.2624511	0.78833413	
0.967827	0.59160346	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x2]

scala> print(output)
(1,.,.) =
0.95979714	0.27654588	
0.35592428	0.49355772	

(2,.,.) =
0.2624511	0.78833413	
0.967827	0.59160346	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2]

```

**Python example:**
```python
model = Clamp(-10, 10)
input = np.random.randn(2, 2, 2)
output = model.forward(input)

>>> print(input)
[[[-0.66763755  1.15392566]
  [-2.10846048  0.46931736]]

 [[ 1.74174638 -1.04323311]
  [-1.91858729  0.12624046]]]
  
>>> print(output)
[[[-0.66763753  1.15392566]
  [-2.10846043  0.46931735]]

 [[ 1.74174643 -1.04323316]
  [-1.91858733  0.12624046]]
```
