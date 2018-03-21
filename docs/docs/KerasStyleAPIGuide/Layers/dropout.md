---
## **Dropout**
Applies Dropout to the input by randomly setting a fraction 'p' of input units to 0 at each update during training time in order to prevent overfitting.

**Scala:**
```scala
Dropout(p, inputShape = null)
```
**Python:**
```python
Dropout(p, input_shape=None)
```

**Parameters:**

* `p`: Fraction of the input units to drop. Double between 0 and 1.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Dropout}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(Dropout(0.3, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-1.1256621	2.5402398	-1.1346831	0.50337905
-1.3835752	0.9513693	-0.24547328	-0.28897092
-0.0302343	-0.4106753	0.46467322	-0.7328933

(2,.,.) =
1.2569109	0.16947697	-0.5000246	2.0856402
-0.04246076	1.5827807	-1.0235463	1.7278075
-0.0035352164	-1.2579697	0.206815	-0.053890422

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
model.add(Dropout(0.8, input_shape=(2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.31594453 0.77216378 0.2613706 ]
  [0.63545339 0.32297202 0.22850705]]

 [[0.19891798 0.52167876 0.94632987]
  [0.56607966 0.37144404 0.37670361]]]
```
Output is
```python
[[[0.0        0.0        1.3068529]
  [0.0        0.0        0.0      ]]

 [[0.0        0.0        4.7316494]
  [0.0        0.0        1.8835181]]]
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
GaussianDropout(p, input_shape=None)
```

**Parameters:**

* `p`: Double, drop probability (as with 'Dropout'). The multiplicative noise will have standard deviation 'sqrt(p/(1-p))'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GaussianDropout}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

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
-0.27553135	0.15290284	-0.23144199	0.619676
-0.6648747	0.053253293	-0.08241931	-0.47651786
-0.46381548	-1.0048811	1.5911313	0.39929882

(2,.,.) =
-0.43828326	0.4397059	-0.7071283	-1.440457
-0.27415445	-1.6525689	-0.14050363	0.8728552
-2.0516112	-2.1537325	1.4714862	0.29218474

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import GaussianDropout

model = Sequential()
model.add(GaussianDropout(0.3, input_shape=(2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.48833477 0.41775547 0.28384793]
  [0.74651192 0.18061384 0.77133968]]

 [[0.50500526 0.38303355 0.9348053 ]
  [0.35643126 0.00460544 0.36511804]]]
```
Output is
```python
[[[0.9491679  0.41856906 0.17491746]
  [0.48217508 0.18805945 0.62253636]]

 [[0.84081125 0.58771574 0.71564806]
  [0.29794857 0.00500419 0.40520114]]]
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
GaussianNoise(sigma, input_shape=None)
```

**Parameters:**

* `sigma`: Double, standard deviation of the noise distribution.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GaussianNoise}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(GaussianNoise(0.6, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.4226985	-0.010519333	-0.49748304	-0.3176052
0.52444375	0.31100306	1.0308859	2.0337727
0.21513703	-0.396619	-0.055275716	-0.40603992

(2,.,.) =
-1.2393064	-0.536477	-0.35633054	-0.09068655
-1.7297741	-0.5812992	-1.2833812	-0.7185058
0.13474904	0.06468039	-0.6630115	1.2471422

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.72299504	0.7733576	-0.13965577	0.72079915
0.20137814	0.6300731	2.5559645	2.3056328
-0.19732013	-0.482926	-0.22114205	-0.88772345

(2,.,.) =
-1.4293398	-1.0870209	-0.5509953	-0.31268832
-2.244024	-0.23773572	-3.022697	-0.65151817
-0.035656676	-0.7470889	-0.8566216	1.1347939

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import GaussianNoise

model = Sequential()
model.add(GaussianNoise(0.45, input_shape=(2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.89283412 0.10962747 0.28798171]
  [0.19249183 0.59853395 0.57247122]]

 [[0.4468411  0.69839321 0.85047587]
  [0.23368985 0.9661946  0.8556528 ]]]
```
Output is
```python
[[[ 0.4730857   0.43211198  0.28431436]
  [-0.75382084  0.69040805  0.74993795]]

 [[-0.00867686  0.6285205   0.79557955]
  [ 0.6101844   0.3889042   0.87428594]]]
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
SpatialDropout1D(p=0.5, input_shape=None)
```

**Parameters:**

* `p`: Fraction of the input units to drop. Double between 0 and 1.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, SpatialDropout1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(SpatialDropout1D(0.45, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.21971664	-1.1752142	1.3990722	0.62127674
0.47111356	-0.2689004	-0.08882793	-2.1752064
0.015462149	0.5628027	-1.2051789	-0.50860554

(2,.,.) =
0.64105046	1.4783055	-1.4207027	0.25074983
-0.4277206	-1.9772152	1.5613455	0.13935184
-0.17331447	1.567511	0.7075385	0.09772631

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.21971664	-1.1752142	0.0	        0.62127674
0.47111356	-0.2689004  -0.0	    -2.1752064
0.015462149	0.5628027	-0.0	    -0.50860554

(2,.,.) =
0.64105046	0.0	        -1.4207027	0.25074983
-0.4277206	-0.0	    1.5613455	0.13935184
-0.17331447	0.0	        0.7075385	0.09772631

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import SpatialDropout1D

model = Sequential()
model.add(SpatialDropout1D(0.3, input_shape=(2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.7404816  0.3750626  0.34632395]
  [0.86605994 0.58908403 0.98963936]]

 [[0.45240542 0.47718322 0.47193128]
  [0.7586274  0.08428707 0.85847428]]]
```
Output is
```python
[[[0.7404816  0.3750626  0.34632394]
  [0.86605996 0.589084   0.98963934]]

 [[0.0        0.47718322 0.47193128]
  [0.0        0.08428707 0.85847425]]]
```

---
## **SpatialDropout2D**
Spatial 2D version of Dropout.

This version performs the same function as Dropout, however it drops entire 2D feature maps instead of individual elements. If adjacent pixels within feature maps are strongly correlated (as is normally the case in early convolution layers) then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, SpatialDropout2D will help promote independence between feature maps and should be used instead.

The input of this layer should be 4D.

**Scala:**
```scala
SpatialDropout2D(p = 0.5, dimOrdering = DataFormat.NCHW, inputShape = null)
```
**Python:**
```python
SpatialDropout2D(p=0.5, dim_ordering="th", input_shape=None)
```

**Parameters:**

* `p`: Fraction of the input units to drop. Double between 0 and 1.
* `dimOrdering`: Format of input data. Either DataFormat.NCHW (dimOrdering='th') or DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, SpatialDropout2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(SpatialDropout2D(0.45, "th", inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.60717005	-0.30501872	0.13144946	-0.14897549
0.9908924	-0.3240618	0.14496706	1.2830952
0.06821448	-1.2557019	0.93521047	-0.32641008

(1,2,.,.) =
-0.8245149	1.5341798	-0.43336067	0.340211
-0.8472337	-0.7606524	0.16646889	1.8325258
2.119133	0.5201503	0.10287065	-0.36848283

(2,1,.,.) =
1.419305	0.42784595	0.21196826	1.1515752
-0.94937575	-0.5575686	1.3476235	2.2079833
0.43185312	0.6149021	-0.92856234	-0.068609774

(2,2,.,.) =
-0.8670012	-0.9826137	0.740112	-0.68443024
-0.2485724	-1.5570805	0.36720234	-0.47003874
1.4752163	-1.3929193	-0.8677884	0.8240998

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.60717005	-0.30501872	0.13144946	-0.14897549
0.9908924	-0.3240618	0.14496706	1.2830952
0.06821448	-1.2557019	0.93521047	-0.32641008

(1,2,.,.) =
-0.8245149	1.5341798	-0.43336067	0.340211
-0.8472337	-0.7606524	0.16646889	1.8325258
2.119133	0.5201503	0.10287065	-0.36848283

(2,1,.,.) =
1.419305	0.42784595	0.21196826	1.1515752
-0.94937575	-0.5575686	1.3476235	2.2079833
0.43185312	0.6149021	-0.92856234	-0.068609774

(2,2,.,.) =
-0.0	    -0.0	    0.0	        -0.0
-0.0	    -0.0	    0.0	        -0.0
0.0	        -0.0	    -0.0	    0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import SpatialDropout2D

model = Sequential()
model.add(SpatialDropout2D(0.3, "tf", input_shape=(2, 3, 4)))
input = np.random.random([1, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.81045721 0.38056422 0.77495337 0.80380938]
   [0.33429186 0.60114337 0.45673243 0.40388212]
   [0.02621719 0.76302171 0.80485438 0.58235836]]

  [[0.55007726 0.3135473  0.31234512 0.50775597]
   [0.48947724 0.171269   0.35376838 0.03825349]
   [0.78897741 0.76452314 0.63108224 0.41506377]]]]
```
Output is
```python
[[[[0.8104572  0.0        0.77495337 0.8038094 ]
   [0.33429185 0.0        0.45673242 0.40388212]
   [0.02621719 0.0        0.8048544  0.58235836]]

  [[0.55007726 0.0        0.31234512 0.507756  ]
   [0.48947725 0.0        0.35376838 0.03825349]
   [0.7889774  0.0        0.63108224 0.41506377]]]]
```

---
## **SpatialDropout3D**
Spatial 3D version of Dropout.

This version performs the same function as Dropout, however it drops entire 3D feature maps instead of individual elements. If adjacent voxels within feature maps are strongly correlated (as is normally the case in early convolution layers) then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, SpatialDropout3D will help promote independence between feature maps and should be used instead.

The input of this layer should be 5D.

**Scala:**
```scala
SpatialDropout3D(p = 0.5, dimOrdering = "CHANNEL_FIRST", inputShape = null)
```
**Python:**
```python
SpatialDropout3D(p=0.5, dim_ordering="th", input_shape=None)
```

**Parameters:**

* `p`: Fraction of the input units to drop. Double between 0 and 1.
* `dimOrdering`: Format of input data. Either 'CHANNEL_FIRST' (dimOrdering='th') or 'CHANNEL_LAST' (dimOrdering='tf'). Default is 'CHANNEL_FIRST'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, SpatialDropout3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(SpatialDropout3D(0.45, "th", inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
-2.117447	 -1.1722422	 -1.0502725
1.1870034	 0.6419669	 0.26606783

(1,1,2,.,.) =
-1.0379035	 -0.10376157 -1.244986
0.32365236	 0.6942544	 1.0216095

(1,2,1,.,.) =
0.99524593	 1.8840269	 -2.0270567
-1.2034	     0.47747976	 0.019542867

(1,2,2,.,.) =
0.07373473	 0.4546184	 1.0401075
-0.41087177	 -1.3756726	 0.18495683

(2,1,1,.,.) =
-1.0301927	 -0.97682196 -0.6294815
0.8123509	 -0.4944477	 -2.5870013

(2,1,2,.,.) =
0.77423155	 -1.1985791	 -0.01225382
-1.840114	 0.24234816	 -1.3665062

(2,2,1,.,.) =
1.3441606	 -0.82009715 -0.916399
0.92800957	 0.43402943	 0.038009416

(2,2,2,.,.) =
-0.12275301	 -1.801634	 -1.0480273
-0.069198474 -0.4556613	 -0.12315401

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
-2.117447	-1.1722422	-1.0502725
1.1870034	0.6419669	0.26606783

(1,1,2,.,.) =
-1.0379035	-0.10376157	-1.244986
0.32365236	0.6942544	1.0216095

(1,2,1,.,.) =
0.99524593	1.8840269	-2.0270567
-1.2034	    0.47747976	0.019542867

(1,2,2,.,.) =
0.07373473	0.4546184	1.0401075
-0.41087177	-1.3756726	0.18495683

(2,1,1,.,.) =
-0.0	    -0.0	    -0.0
0.0	        -0.0	    -0.0

(2,1,2,.,.) =
0.0	        -0.0	    -0.0
-0.0	    0.0	        -0.0

(2,2,1,.,.) =
0.0	        -0.0	    -0.0
0.0	        0.0	        0.0

(2,2,2,.,.) =
-0.0	    -0.0	    -0.0
-0.0	    -0.0	    -0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import SpatialDropout3D

model = Sequential()
model.add(SpatialDropout3D(0.3, "tf", input_shape=(2, 2, 2, 2)))
input = np.random.random([1, 2, 2, 2, 2])
output = model.forward(input)
```
Input is:
```python
[[[[[0.31925732 0.16270694]
    [0.72014256 0.54190354]]

   [[0.60849703 0.08263826]
    [0.15298021 0.17085947]]]


  [[[0.32196594 0.84327244]
    [0.17249493 0.86023903]]

   [[0.89705676 0.02341975]
    [0.86830273 0.65951661]]]]]
```
Output is
```python
[[[[[0.0        0.16270694]
    [0.0        0.54190356]]

   [[0.0        0.08263826]
    [0.0        0.17085947]]]


  [[[0.0        0.84327245]
    [0.0        0.860239  ]]

   [[0.0        0.02341975]
    [0.0        0.65951663]]]]]

```

---
