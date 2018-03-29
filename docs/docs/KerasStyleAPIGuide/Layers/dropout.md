## **Dropout**
Applies Dropout to the input by randomly setting a fraction 'p' of input units to 0 at each update during training time in order to prevent overfitting.

**Scala:**
```scala
Dropout(p, inputShape = null)
```
**Python:**
```python
Dropout(p, input_shape=None, name=None)
```

**Parameters:**

* `p`: Fraction of the input units to drop. Between 0 and 1.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Dropout}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Dropout(0.3, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-1.1256621	   2.5402398	-1.1346831	 0.50337905
-1.3835752	   0.9513693	-0.24547328	 -0.28897092
-0.0302343	   -0.4106753	0.46467322	 -0.7328933

(2,.,.) =
1.2569109	   0.16947697	-0.5000246	 2.0856402
-0.04246076	   1.5827807	-1.0235463	 1.7278075
-0.0035352164  -1.2579697	0.206815	 -0.053890422

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.0	          0.0	     -1.620976	0.0
-1.976536	  1.359099	 0.0	    -0.4128156
-0.043191858  -0.586679	 0.6638189	-1.0469904

(2,.,.) =
0.0	           0.0	       -0.7143209  2.979486
-0.060658228   2.2611153   0.0	       0.0
-0.0050503095  -1.7970997  0.0	       0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Dropout

model = Sequential()
model.add(Dropout(0.3, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.61976372 0.36074095 0.59003926 0.75373888]
  [0.2390103  0.93491731 0.89078166 0.93083315]
  [0.62360382 0.73646417 0.32886041 0.25372008]]

 [[0.10235195 0.7782206  0.54940485 0.41757437]
  [0.94804637 0.04642807 0.17194449 0.2675274 ]
  [0.89322413 0.3301816  0.49910094 0.00819342]]]
```
Output is
```python
[[[0.88537675 0.51534426 0.0        0.0       ]
  [0.0        1.3355962  1.2725452  1.3297616 ]
  [0.89086264 1.0520917  0.4698006  0.0       ]]

 [[0.14621708 1.1117437  0.7848641  0.59653485]
  [1.354352   0.06632582 0.24563499 0.382182  ]
  [1.2760345  0.471688   0.7130013  0.01170488]]]
```

---
## **GaussianDropout**
Apply multiplicative 1-centered Gaussian noise.

As it is a regularization layer, it is only active at training time.

**Scala:**
```scala
GaussianDropout(p, inputShape = null)
```
**Python:**
```python
GaussianDropout(p, input_shape=None, name=None)
```

**Parameters:**

* `p`: Drop probability (as with 'Dropout'). The multiplicative noise will have standard deviation 'sqrt(p/(1-p))'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GaussianDropout}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GaussianDropout(0.45, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.14522108	0.27993536	-0.38809696	0.26372102
-0.5572615	0.091684595	0.27881327	-1.6235427
-0.32884964	-0.46456075	1.6169231	0.31943536

(2,.,.) =
-1.813811	1.1577623	-0.8995344	-1.0607182
-0.3952898	-2.3437335	-0.6608733	1.1752778
-1.3373735	-1.7404749	0.82832927	0.3053458

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.27553135	 0.15290284	  -0.23144199  0.619676
-0.6648747	 0.053253293  -0.08241931  -0.47651786
-0.46381548	 -1.0048811	  1.5911313	   0.39929882

(2,.,.) =
-0.43828326	 0.4397059	  -0.7071283   -1.440457
-0.27415445	 -1.6525689	  -0.14050363  0.8728552
-2.0516112	 -2.1537325	  1.4714862	   0.29218474

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import GaussianDropout

model = Sequential()
model.add(GaussianDropout(0.45, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.87208899 0.1353189  0.325058   0.63174633]
  [0.20479221 0.29774652 0.42038452 0.23819006]
  [0.07608872 0.91696766 0.245824   0.84324374]]

 [[0.26268714 0.76275494 0.63620997 0.15049668]
  [0.54144135 0.70412821 0.05555471 0.72317157]
  [0.32796076 0.26804862 0.80775221 0.46948471]]]
```
Output is
```python
[[[ 2.1392     -0.16185573 -0.18517245 -0.36539674]
  [ 0.15324984  0.17320508  0.82520926  0.21734479]
  [ 0.17601383  0.24906069  0.15664667  0.12675671]]

 [[ 0.49689308  1.8231225   1.0023257   0.37604305]
  [ 1.2827866  -0.08726044  0.01333602  0.8518126 ]
  [ 0.20021693  0.31828243  1.0940336   0.00866747]]]
```

---
## **GaussianNoise**
Apply additive zero-centered Gaussian noise.

This is useful to mitigate overfitting (you could see it as a form of random data augmentation).

Gaussian Noise is a natural choice as corruption process for real valued inputs.

As it is a regularization layer, it is only active at training time.

**Scala:**
```scala
GaussianNoise(sigma, inputShape = null)
```
**Python:**
```python
GaussianNoise(sigma, input_shape=None, name=None)
```

**Parameters:**

* `sigma`: Standard deviation of the noise distribution.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GaussianNoise}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GaussianNoise(0.6, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.4226985	-0.010519333  -0.49748304	-0.3176052
0.52444375	0.31100306	  1.0308859	    2.0337727
0.21513703	-0.396619	  -0.055275716	-0.40603992

(2,.,.) =
-1.2393064	-0.536477	  -0.35633054	-0.09068655
-1.7297741	-0.5812992	  -1.2833812	-0.7185058
0.13474904	0.06468039	  -0.6630115	1.2471422

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.72299504	  0.7733576	   -0.13965577	 0.72079915
0.20137814	  0.6300731	   2.5559645	 2.3056328
-0.19732013	  -0.482926	   -0.22114205	 -0.88772345

(2,.,.) =
-1.4293398	  -1.0870209   -0.5509953	 -0.31268832
-2.244024	  -0.23773572  -3.022697	 -0.65151817
-0.035656676  -0.7470889   -0.8566216	 1.1347939

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import GaussianNoise

model = Sequential()
model.add(GaussianNoise(0.6, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.61699657 0.9759922  0.62898391 0.57265605]
  [0.88815108 0.9484446  0.0300381  0.54114527]
  [0.94046216 0.05998474 0.24860526 0.82020617]]

 [[0.87308242 0.24780141 0.73385444 0.40836049]
  [0.33166358 0.74540915 0.28333526 0.08263288]
  [0.17527315 0.79798327 0.49351559 0.13895365]]]
```
Output is
```python
[[[ 1.5833025   1.1431103   0.14338043  1.634818  ]
  [ 0.01713479  1.1608562   0.222246    0.40559798]
  [ 0.9930201   0.1187391   -0.35643864 -0.7164774 ]]

 [[ 1.0105296   1.423961    0.90040827  1.3460591 ]
  [ 0.943779    -0.48430538 0.20670155  -0.50143087]
  [ -0.29849088 0.12774569  -0.16126743 -0.011041  ]]]
```

---
## **SpatialDropout1D**
Spatial 1D version of Dropout.

This version performs the same function as Dropout, however it drops entire 1D feature maps instead of individual elements. If adjacent frames within feature maps are strongly correlated (as is normally the case in early convolution layers) then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, SpatialDropout1D will help promote independence between feature maps and should be used instead.

The input of this layer should be 3D.

**Scala:**
```scala
SpatialDropout1D(p = 0.5, inputShape = null)
```
**Python:**
```python
SpatialDropout1D(p=0.5, input_shape=None, name=None)
```

**Parameters:**

* `p`: Fraction of the input units to drop. Between 0 and 1.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, SpatialDropout1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(SpatialDropout1D(inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.41961443	-0.3900255	-0.11937201	1.2904007
-1.7623849	-1.6778483	-0.30053464	0.33295104
-0.29824665	-0.25474855	-2.1878588	1.2741995

(2,.,.) =
0.24517925	2.0451863	-0.4281332	-1.2022524
-0.7767442	0.24794191	-0.5614063	0.14720131
-1.4832486	0.59478635	-0.13351384	-0.8799204

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.41961443	-0.0   -0.11937201	1.2904007
-1.7623849	-0.0   -0.30053464	0.33295104
-0.29824665	-0.0   -2.1878588	1.2741995

(2,.,.) =
0.24517925	0.0	   -0.4281332	-0.0
-0.7767442	0.0	   -0.5614063	0.0
-1.4832486	0.0	   -0.13351384	-0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import SpatialDropout1D

model = Sequential()
model.add(SpatialDropout1D(input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.67162434 0.91104925 0.66869854 0.17295748]
  [0.78326617 0.27447329 0.18051406 0.24230118]
  [0.7098933  0.32496974 0.00517668 0.21293476]]

 [[0.26932307 0.33496273 0.71258256 0.15464896]
  [0.75286915 0.210486   0.91826256 0.81379954]
  [0.11960744 0.37420041 0.03886506 0.22882457]]]
```
Output is
```python
[[[0.0        0.0        0.0        0.0       ]
  [0.0        0.0        0.0        0.0       ]
  [0.0        0.0        0.0        0.0       ]]

 [[0.0        0.33496273 0.0        0.15464896]
  [0.0        0.210486   0.0        0.81379956]
  [0.0        0.3742004  0.0        0.22882457]]]
```

---
## **SpatialDropout2D**
Spatial 2D version of Dropout.

This version performs the same function as Dropout, however it drops entire 2D feature maps instead of individual elements. If adjacent pixels within feature maps are strongly correlated (as is normally the case in early convolution layers) then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, SpatialDropout2D will help promote independence between feature maps and should be used instead.

The input of this layer should be 4D.

**Scala:**
```scala
SpatialDropout2D(p = 0.5, dimOrdering = "th", inputShape = null)
```
**Python:**
```python
SpatialDropout2D(p=0.5, dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `p`: Fraction of the input units to drop. Between 0 and 1.
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, SpatialDropout2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(SpatialDropout2D(inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
1.1651757	1.0867785	-0.56122786	-0.542156
-0.79321486	0.64733976	-0.7040698	-0.8619171
-0.61122066	-1.9640825	-1.0078672	-0.12195914

(1,2,.,.) =
-0.24738677	-0.9351172	-0.11694977	0.8657273
-0.4773825	-1.6853696	-1.4906564	-0.06981948
-0.8184341	-1.3537912	1.2442955	-0.0071462104

(2,1,.,.) =
1.8801081	0.44946647	0.47776535	0.036228795
-1.2122079	0.41413695	-0.691067	2.6273472
1.4293005	-1.2627622	-1.8263477	0.015581204

(2,2,.,.) =
2.0050068	-0.32893315	0.19670151	0.8031714
0.16645809	-0.68172836	0.5169275	-0.83938134
0.1789333	2.1845143	1.3843338	-0.8283524

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.0	        0.0	        -0.0	    -0.0
-0.0	    0.0	        -0.0	    -0.0
-0.0	    -0.0	    -0.0	    -0.0

(1,2,.,.) =
-0.0	    -0.0	    -0.0	    0.0
-0.0	    -0.0	    -0.0	    -0.0
-0.0	    -0.0	    0.0	        -0.0

(2,1,.,.) =
1.8801081	0.44946647	0.47776535	0.036228795
-1.2122079	0.41413695	-0.691067	2.6273472
1.4293005	-1.2627622	-1.8263477	0.015581204

(2,2,.,.) =
2.0050068	-0.32893315	0.19670151	0.8031714
0.16645809	-0.68172836	0.5169275	-0.83938134
0.1789333	2.1845143	1.3843338	-0.8283524

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import SpatialDropout2D

model = Sequential()
model.add(SpatialDropout2D(input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.21864846 0.43531162 0.23078088 0.81122115]
   [0.19442596 0.11110444 0.533805   0.68291312]
   [0.40738259 0.05448269 0.04647733 0.41683944]]
  [[0.23354645 0.46005503 0.87695602 0.13318982]
   [0.2596346  0.67654484 0.79389709 0.50408343]
   [0.50043622 0.28028835 0.81897585 0.01629935]]]

 [[[0.32173241 0.38367311 0.10315543 0.22691558]
   [0.41640003 0.45932496 0.70795718 0.67185326]
   [0.11911477 0.90231481 0.49881045 0.74297438]]
  [[0.48873758 0.53475116 0.06801025 0.50640297]
   [0.95740488 0.14928652 0.10466387 0.29040436]
   [0.44062539 0.36983024 0.35326756 0.60592402]]]]
```
Output is
```python
[[[[0.21864846 0.43531162 0.23078088 0.8112211 ]
   [0.19442596 0.11110444 0.533805   0.6829131 ]
   [0.4073826  0.05448269 0.04647733 0.41683942]]
  [[0.23354645 0.46005502 0.87695605 0.13318983]
   [0.2596346  0.67654485 0.7938971  0.50408345]
   [0.50043625 0.28028834 0.81897587 0.01629935]]]

 [[[0.0        0.0        0.0        0.0       ]
   [0.0        0.0        0.0        0.0       ]
   [0.0        0.0        0.0        0.0       ]]
  [[0.48873758 0.5347512  0.06801025 0.50640297]
   [0.95740485 0.14928652 0.10466387 0.29040435]
   [0.4406254  0.36983025 0.35326755 0.605924  ]]]]
```

---
## **SpatialDropout3D**
Spatial 3D version of Dropout.

This version performs the same function as Dropout, however it drops entire 3D feature maps instead of individual elements. If adjacent voxels within feature maps are strongly correlated (as is normally the case in early convolution layers) then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, SpatialDropout3D will help promote independence between feature maps and should be used instead.

The input of this layer should be 5D.

**Scala:**
```scala
SpatialDropout3D(p = 0.5, dimOrdering = "th", inputShape = null)
```
**Python:**
```python
SpatialDropout3D(p=0.5, dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `p`: Fraction of the input units to drop. Between 0 and 1.
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, SpatialDropout3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(SpatialDropout3D(inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
0.28834015	  -0.74598366  0.16951436
0.17009573	  0.3626017	   -0.24652131

(1,1,2,.,.) =
1.3008109	  0.37243804   0.073205866
1.0715603	  0.02033514   -1.7862324

(1,2,1,.,.) =
-0.5285066	  -1.3859391   -1.0543352
0.7904896	  0.7473174	   -0.5941196

(1,2,2,.,.) =
-0.060706574  -2.4405587	1.5963978
-0.33285397	  -0.48576602	0.8121179

(2,1,1,.,.) =
-0.7060156	  0.31667668	-0.28765643
-1.3115436	  -1.7266335	1.0080509

(2,1,2,.,.) =
1.2365453	  -0.13272893	-1.2130978
0.26921487	  -0.66259027	0.5537464

(2,2,1,.,.) =
1.6578121	  -0.09890133	0.4794677
1.5102282	  0.067802615	0.76998603

(2,2,2,.,.) =
-0.47348467	  0.19535838	0.62601316
-2.4771519	  -0.40744382	0.04029308

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
0.0	            -0.0	    0.0
0.0	            0.0	        -0.0

(1,1,2,.,.) =
0.0	            0.0	        0.0
0.0	            0.0	        -0.0

(1,2,1,.,.) =
-0.5285066	    -1.3859391	-1.0543352
0.7904896	    0.7473174	-0.5941196

(1,2,2,.,.) =
-0.060706574	-2.4405587	1.5963978
-0.33285397	    -0.48576602	0.8121179

(2,1,1,.,.) =
-0.0	        0.0	        -0.0
-0.0	        -0.0	    0.0

(2,1,2,.,.) =
0.0	            -0.0	    -0.0
0.0	            -0.0	    0.0

(2,2,1,.,.) =
0.0	            -0.0	    0.0
0.0	            0.0	        0.0

(2,2,2,.,.) =
-0.0	        0.0	        0.0
-0.0	        -0.0	    0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import SpatialDropout3D

model = Sequential()
model.add(SpatialDropout3D(input_shape=(2, 2, 2, 2)))
input = np.random.random([2, 2, 2, 2, 2])
output = model.forward(input)
```
Input is:
```python
[[[[[0.68128454 0.57379206]
    [0.19533742 0.19906853]]
   [[0.21527836 0.79586573]
    [0.51065215 0.94422278]]]
  [[[0.95178211 0.50359204]
    [0.0306965  0.92563536]]
   [[0.33744311 0.58750719]
    [0.45437398 0.7081438 ]]]]

 [[[[0.00235233 0.8092749 ]
    [0.65525661 0.01079958]]
   [[0.29877429 0.42090468]
    [0.28265598 0.81520172]]]
  [[[0.91811333 0.3275563 ]
    [0.66125455 0.15555596]]
   [[0.53651033 0.66013486]
    [0.45874838 0.7613676 ]]]]]
```
Output is
```python
[[[[[0.68128455 0.57379204]
    [0.19533743 0.19906853]]
   [[0.21527836 0.7958657 ]
    [0.5106521  0.94422275]]]
  [[[0.0        0.0       ]
    [0.0        0.0       ]]
   [[0.0        0.0       ]
    [0.0        0.0       ]]]]

 [[[[0.0        0.0       ]
    [0.0        0.0       ]]
   [[0.0        0.0       ]
    [0.0        0.0       ]]]
  [[[0.91811335 0.3275563 ]
    [0.6612545  0.15555596]]
   [[0.53651035 0.66013485]
    [0.45874837 0.7613676 ]]]]]
```
