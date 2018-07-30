## **MaxPooling1D**
Max pooling operation for temporal data.

Input shape:

3D tensor with shape: `(batch_size, steps, features)`.

Output shape:

3D tensor with shape: `(batch_size, downsampled_steps, features)`.

**Scala:**
```scala
MaxPooling1D(poolSize = 2, strides = -1, padding = "valid", inputShape = null)
```
**Python:**
```python
MaxPooling1D(pool_size=2, strides=None, padding="valid", input_shape=None, name=None)
```

Parameters:

* `pool_size`: Integer, size of the max pooling windows. Default is 2.
* `strides`: Integer, or None. Factor by which to downscale.
            E.g. 2 will halve the input.
            If None, it will be set to -1, which will be default to pool_size.
* `padding`: One of `"valid"` or `"same"` (case-insensitive).
* `input_shape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a shape object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras2.layers.MaxPooling1D

val seq = Sequential[Float]()
val layer = MaxPooling1D[Float](poolSize = 3, inputShape = Shape(4, 5))
seq.add(layer)
val input = Tensor[Float](3, 4, 5).randn()
val output = seq.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.8663151	-0.23868443	-0.00667114	0.57754266	-1.1602311	
1.453712	0.1588971	0.71542656	1.6594391	0.04762305	
2.6128695	-0.4991936	0.37722582	0.07411593	1.8921419	
0.85887206	0.99118704	0.717694	0.5889265	1.4998609	

(2,.,.) =
-2.3484302	-0.40476575	0.29646426	0.6123499	-2.3566647	
-0.27840275	-0.0931928	-1.0732254	-0.28036273	-0.19488569	
0.4438278	-0.13150546	-0.8513687	0.5319984	1.4565476	
-1.8620179	-0.19861892	-0.14878958	1.4498321	0.016972434	

(3,.,.) =
-1.6097814	1.1128851	-1.7357148	-0.3583554	-0.6089424	
2.1183956	0.6400526	0.26053894	-2.5416205	-0.9832211	
0.2170545	0.28168106	-0.009057811	1.7110301	1.0579157	
-0.46720266	-0.87794846	0.2708433	-1.6016585	0.23714945	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4x5]
```
Output is:
```scala
output: 
(1,.,.) =
2.6128695	0.1588971	0.71542656	1.6594391	1.8921419	

(2,.,.) =
0.4438278	-0.0931928	0.29646426	0.6123499	1.4565476	

(3,.,.) =
2.1183956	1.1128851	0.26053894	1.7110301	1.0579157	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x1x5]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras2.layers import MaxPooling1D

model = Sequential()
model.add(MaxPooling1D(pool_size=3, input_shape=(4, 5)))
input = np.random.random([3, 4, 5])
output = model.forward(input)
```
Input is:
```python
[[[0.65224256 0.20913976 0.37685946 0.57730155 0.66970161]
  [0.26367096 0.94933166 0.09015655 0.15487805 0.74161953]
  [0.55753943 0.6805039  0.42343199 0.25051912 0.59142363]
  [0.77773199 0.57204293 0.22107977 0.54485067 0.00167209]]

 [[0.12947324 0.13329145 0.95614414 0.85027407 0.5055782 ]
  [0.40469384 0.46683239 0.57578244 0.67001174 0.53348182]
  [0.2170539  0.51992802 0.31005515 0.67494018 0.74330305]
  [0.48408556 0.26271    0.32412418 0.08973007 0.01880989]]

 [[0.80010078 0.54220073 0.653223   0.29034995 0.11341325]
  [0.66065103 0.49484952 0.83346121 0.47914374 0.81174956]
  [0.29151878 0.61798409 0.74534208 0.92317947 0.54144718]
  [0.26080681 0.41200147 0.79630472 0.6739419  0.435016  ]]]
```
Output is:
```python
[[[0.65224254 0.94933164 0.423432   0.57730156 0.7416195 ]]

 [[0.40469384 0.51992804 0.95614415 0.8502741  0.74330306]]

 [[0.8001008  0.6179841  0.8334612  0.92317945 0.8117496 ]]]
```

---
## **AveragePooling1D**
Average pooling for temporal data.

Input shape:

3D tensor with shape: `(batch_size, steps, features)`.

Output shape:

3D tensor with shape: `(batch_size, downsampled_steps, features)`.

**Scala:**
```scala
AveragePooling1D(poolSize = 2, strides = -1, padding = "valid", inputShape = null)
```
**Python:**
```python
AveragePooling1D(pool_size=2, strides=None, padding="valid", input_shape=None, name=None)
```

Parameters:

* `pool_size`: Integer, size of the average pooling windows. Default is 2.
* `strides`: Integer, or None. Factor by which to downscale.
            E.g. 2 will halve the input.
            If None, it will be set to -1, which will be default to pool_size.
* `padding`: One of `"valid"` or `"same"` (case-insensitive).
* `input_shape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a shape object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras2.layers.AveragePooling1D

 val seq = Sequential[Float]()
val layer = AveragePooling1D[Float](poolSize = 3, inputShape = Shape(4, 5))
seq.add(layer)
val input = Tensor[Float](3, 4, 5).randn()
val output = seq.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.7240005	-0.06335294	0.6269808	0.24385877	0.014416845	
-0.7663384	0.58187956	0.35314906	0.60641724	-0.2761965	
1.0691774	-0.671351	0.7121989	0.7686876	0.14512675	
-0.83099926	-0.31942126	-0.8608722	1.3061627	0.67469263	

(2,.,.) =
0.32662675	-0.8206316	-2.312093	-1.2558469	0.0048087407	
-0.46804208	-0.008146223	-1.3610557	-0.29545167	-0.9627323	
-0.04214055	0.073838815	0.018005485	0.22931503	-0.6118381	
0.23300731	0.059008796	-0.58128744	-0.49869594	0.6242729	

(3,.,.) =
0.60912937	0.6315228	-0.23742959	-1.1818335	0.11456228	
-0.4489492	0.2996443	-0.9002065	-0.53337836	-0.019300539	
0.2350515	-1.0584278	-0.051998064	-1.2586755	0.17742781	
1.3912749	-0.97213763	1.573626	0.36101308	0.11868247	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4x5]
```
Output is:
```scala
output: 
(1,.,.) =
0.67561316	-0.050941467	0.56410956	0.53965455	-0.03888431	

(2,.,.) =
-0.06118529	-0.25164634	-1.218381	-0.44066116	-0.5232539	

(3,.,.) =
0.1317439	-0.04242025	-0.3965447	-0.9912958	0.09089652	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x1x5]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras2.layers import AveragePooling1D

model = Sequential()
model.add(AveragePooling1D(pool_size=3, input_shape=(4, 5)))
input = np.random.random([3, 4, 5])
output = model.forward(input)
```
Input is:
```python
[[[0.52804108 0.69499686 0.42818504 0.2107447  0.39879583]
  [0.01134639 0.74244503 0.71352988 0.4656967  0.56990969]
  [0.60781477 0.34275853 0.09997229 0.15758047 0.38226729]
  [0.8884477  0.57125125 0.19161781 0.70570558 0.38504297]]

 [[0.68505125 0.10344407 0.26294358 0.84476828 0.37145984]
  [0.5073895  0.17534648 0.88501456 0.12600059 0.43856957]
  [0.74562707 0.25003322 0.34860707 0.16645732 0.04184937]
  [0.5352171  0.85188837 0.64421649 0.51544795 0.7619103 ]]

 [[0.08110649 0.84742271 0.08083711 0.43689189 0.21256946]
  [0.24837393 0.53503375 0.41418659 0.34652157 0.00923598]
  [0.89420575 0.60971584 0.7718259  0.06192155 0.09029334]
  [0.72170843 0.0589505  0.78960517 0.9543299  0.2462495 ]]]
```
Output is:
```python
[[[0.38240075 0.5934002  0.41389573 0.2780073  0.4503243 ]]

 [[0.6460226  0.17627458 0.49885502 0.3790754  0.2839596 ]]

 [[0.40789542 0.66405743 0.4222832  0.2817783  0.10403293]]]
```
