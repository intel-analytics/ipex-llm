## Bottle ##

**Scala:**
```scala
val model = Bottle[T](module, nInputDim, nOutputDim)
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

val model = Bottle[Float](Linear[Float](10, 2), 2, 2)
model.add(Linear(10, 2))
val input = Tensor[Float](4, 5, 10).rand()
val output = model.forward(input)
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
(1,.,.) =
0.6945294	-0.27953064	
0.6268389	-0.7294409	
0.69834805	-0.42664433	
0.70373046	-0.4026499	
0.66308194	-0.6336497	

(2,.,.) =
0.76823425	-0.57179654	
0.54741347	-0.5171715	
0.6170485	-0.48814133	
0.89729875	-0.5363091	
0.9383141	-0.63053	

(3,.,.) =
0.6869495	-0.6013391	
0.72504604	-0.44045419	
0.84359026	-0.51410943	
0.7153435	-0.783236	
0.8234116	-0.6176827	

(4,.,.) =
0.8869035	-0.51233184	
0.65199244	-0.48857856	
0.7880871	-0.7456757	
0.8663832	-0.22757408	
0.9411352	-0.8008182	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 4x5x2]
```

**Python example:**
```python
model = Bottle(Linear(10, 2), 2, 2)
model.add(Linear(10, 2))

input = np.random.randn(4, 5, 10)
output = model.forward(input)
```
output is
```
array([[[-0.27106649,  0.14462236],
        [-0.28050202,  0.52495712],
        [-0.3351084 ,  1.02605069],
        [-0.18306208,  0.17746207],
        [ 0.62624407, -0.01642358]],

       [[-0.49295783,  0.58246708],
        [ 0.25713843,  0.69936216],
        [ 0.42159486,  0.89629161],
        [-1.32501745, -0.010911  ],
        [-0.06844211,  0.38141996]],

       [[-0.60752791, -0.01978694],
        [ 0.58144319,  1.03134239],
        [ 0.48065221, -0.86170584],
        [-0.51053059,  1.06431556],
        [ 0.46424878, -0.29169011]],

       [[-0.00934486,  0.68141061],
        [ 0.15537724, -0.09281653],
        [-0.23529468,  0.78246176],
        [-0.348032  , -0.40045774],
        [-1.087569  ,  1.19862282]]], dtype=float32)
```
