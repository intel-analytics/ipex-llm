## SoftPlus ##

**Scala:**
```scala
val model = SoftPlus[T]()
```
**Python:**
```python
model = SoftPlus()
```

Apply the SoftPlus function to an n-dimensional input tensor.
SoftPlus function: 
```
f_i(x) = 1/beta * log(1 + exp(beta * x_i))
```
- param beta Controls sharpness of transfer function

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

val model = SoftPlus[Float]()
val input = Tensor[Float](2, 3, 4).rand()
val output = model.forward(input)
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
(1,.,.) =
0.7835901	0.9962422	0.7521913	1.1398541	
0.71477914	1.0578016	0.7928455	0.6975596	
0.8969599	0.88751	0.7120005	0.9749961	

(2,.,.) =
1.0419492	0.76816565	1.0519257	1.1393704	
0.72922856	1.1668311	0.7839851	0.94722736	
0.8995574	1.1064267	1.1123171	0.92133987	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
**Python example:**
```python
model = SoftPlus()
input = np.random.randn(2, 3, 4)
output = model.forward(input)
```
output is
```
array([[[ 0.09477428,  1.28491187,  0.65586591,  1.1293689 ],
        [ 0.37589449,  0.71724343,  0.31651947,  0.84333116],
        [ 0.11157955,  0.91336811,  0.8104986 ,  1.13143706]],

       [[ 0.59171015,  0.20237219,  2.18983054,  0.8992095 ],
        [ 0.38286343,  0.14144027,  0.28824955,  1.80149364],
        [ 0.46423054,  0.52238309,  1.34621668,  1.61121106]]], dtype=float32)
```
