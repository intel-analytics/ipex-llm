## Clamp ##

**Scala:**
```scala
val model = Clamp[T](min, max)
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
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

val model = Clamp[Float](-10, 10)
val input = Tensor[Float](2, 2, 2).rand()
val output = model.forward(input)
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
(1,.,.) =
0.19594982	0.1558478	
0.23255411	0.8538258	

(2,.,.) =
0.76815903	0.0132634975	
0.33081427	0.5836359	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2]
```

**Python example:**
```python
model = Clamp(-10, 10)
input = np.random.randn(2, 2, 2)
output = model.forward(input)
```
output is
```
array([[[ 0.01126319,  0.02390726],
        [-1.15782905, -0.36142176]],

       [[-2.31166029,  1.21416366],
        [-0.28188094, -0.24606584]]], dtype=float32)
```
