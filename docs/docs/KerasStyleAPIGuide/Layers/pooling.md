---
## **MaxPooling1D**
Max pooling operation for temporal data.

The input of this layer should be 3D.

**Scala:**
```scala
MaxPooling1D(poolLength = 2, stride = -1, borderMode = "valid", inputShape = null)
```
**Python:**
```python
MaxPooling1D(pool_length=2, stride=None, border_mode="valid", input_shape=None)
```

Parameters:

* `poolLength`: Size of the region to which max pooling is applied.
* `stride`: Factor by which to downscale. Integer, or -1. 2 will halve the input. If -1, it will default to poolLength. Default is -1.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, MaxPooling1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(MaxPooling1D(poolLength = 3, inputShape = Shape(4, 5)))
val input = Tensor[Float](3, 4, 5).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.32697344	 -1.901702	  0.9338836	   0.4988416	-1.4769285
0.82112324	 -1.749153	  -1.2225364   0.17479241	-0.1569928
-1.9349245	 -0.7208759	  -2.6186085   0.7094514	0.02309827
0.06299127	 -0.28094748  -1.679667	   -0.19593267	-0.6486389

(2,.,.) =
0.5059762	 -0.27661985  1.3978469	   -0.13661754	0.9121702
1.20289	     -1.2779995	  -1.221474	   1.6933655	0.06884759
-0.8358409	 -1.5242177	  0.38067985   0.1758138	-2.0869224
-0.052700672 -1.2065598	  0.65831304   -2.7004414	-1.5840155

(3,.,.) =
-1.5877407	 -0.23685509  -1.1487285   0.6082965	0.5463596
-0.6323151	 1.6099663	  0.16473362   -0.6759079	-0.22952202
0.07198518	 1.0313594	  1.4555247	   0.7538992	-1.2048378
1.2034347	 0.11312642	  -0.14845283  -1.3795642	1.1672769

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4x5]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.82112324	-0.7208759	0.9338836	0.7094514	0.02309827

(2,.,.) =
1.20289	    -0.27661985	1.3978469	1.6933655	0.9121702

(3,.,.) =
0.07198518	1.6099663	1.4555247	0.7538992	0.5463596

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x1x5]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import MaxPooling1D

model = Sequential()
model.add(MaxPooling1D(pool_length = 3, input_shape = (4, 5)))
input = np.random.random([3, 4, 5])
output = model.forward(input)
```
Input is:
```python
[[[0.40580359 0.91869648 0.52699134 0.96507862 0.45316868]
  [0.55665601 0.91599093 0.68640946 0.55788983 0.79788871]
  [0.63706076 0.86559853 0.2157637  0.56051023 0.48453306]
  [0.68673896 0.35445905 0.98369363 0.05747027 0.54176785]]

 [[0.00154654 0.02109022 0.69103023 0.08356977 0.51230376]
  [0.01498106 0.32251403 0.98859889 0.6393191  0.59248678]
  [0.43467219 0.97269656 0.82172126 0.62731276 0.19477236]
  [0.44162847 0.50752131 0.43099026 0.07546448 0.97122237]]

 [[0.9526254  0.82221173 0.13355431 0.19929353 0.95937559]
  [0.53449677 0.8041899  0.45077759 0.40048272 0.31712774]
  [0.83603459 0.72547619 0.61066729 0.09561956 0.32530191]
  [0.10199395 0.77512743 0.69522612 0.7456257  0.73544269]]]
```
Output is:
```python
[[[0.63706076 0.91869646 0.6864095  0.9650786  0.7978887 ]]

 [[0.43467218 0.97269654 0.9885989  0.6393191  0.5924868 ]]

 [[0.9526254  0.82221174 0.6106673  0.4004827  0.95937556]]]
```

---
## **MaxPooling2D**
Max pooling operation for spatial data.

The input of this layer should be 4D.

**Scala:**
```scala
MaxPooling2D(poolSize = Array(2, 2), strides = null, borderMode = "valid", dimOrdering = DataFormat.NCHW, inputShape = null)
```
**Python:**
```python
MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th', input_shape=None)
```

Parameters:

* `poolSize`: Int array of length 2 corresponding to the downscale vertically and horizontally. Default is (2, 2), which will halve the image in each dimension.
* `strides`: Int array of length 2. Stride values. Default is null, and in this case it will be equal to poolSize.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `dimOrdering`: Format of input data. Either DataFormat.NCHW (dimOrdering='th') or DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, MaxPooling2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(MaxPooling2D(inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
2.5301383	0.10926374	0.6072471	0.658932
-0.3569041	0.32731345	-1.4209954	-0.4969882
0.70455354	-2.7349844	0.66514283	-1.0055662

(1,2,.,.) =
-0.29669985	0.054489832	-1.1771511	-0.37510478
1.2857671	-1.1703448	0.39755398	-1.6102049
-0.42201662	1.2561954	1.1706035	0.20676066

(2,1,.,.) =
2.2395058	0.36936793	-1.0407287	0.46479732
0.08024679	-1.3457166	-0.7048267	-0.017787607
-0.66454273	-1.5704913	-1.7375602	-2.417642

(2,2,.,.) =
-1.5279706	-1.0108438	1.0017345	-0.5810244
-1.5944351	0.11111861	0.4439802	-0.48056543
-2.4090567	-1.459287	0.67291117	0.24757418

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
2.5301383	0.658932

(1,2,.,.) =
1.2857671	0.39755398

(2,1,.,.) =
2.2395058	0.46479732

(2,2,.,.) =
0.11111861	1.0017345

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x1x2]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=(2, 1), input_shape = (2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.32958058 0.72913503 0.55085989 0.10701207]
   [0.85694579 0.71180082 0.91934561 0.88084285]
   [0.11031712 0.6436052  0.25791524 0.54819584]]

  [[0.19538002 0.94335506 0.18044102 0.97288022]
   [0.35145474 0.04904515 0.70007689 0.66552503]
   [0.69555239 0.46078683 0.50545691 0.49554022]]]


 [[[0.20044618 0.6763749  0.14910154 0.13245415]
   [0.31930028 0.97913256 0.06482784 0.77955057]
   [0.38687166 0.30410623 0.58693189 0.74744312]]

  [[0.66011955 0.5426731  0.41506221 0.20697936]
   [0.75554922 0.5796121  0.67609129 0.16380068]
   [0.83396812 0.92883048 0.10789639 0.22041409]]]]
```
Output is:
```python
[[[[0.8569458  0.72913504 0.9193456  0.88084286]]

  [[0.35145473 0.9433551  0.7000769  0.97288024]]]


 [[[0.31930026 0.97913253 0.14910154 0.77955055]]

  [[0.75554925 0.5796121  0.6760913  0.20697936]]]]
```

---
## **MaxPooling3D**
Applies max pooling operation for 3D data (spatial or spatio-temporal).

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

Border mode currently supported for this layer is 'valid'.

The input of this layer should be 5D.

**Scala:**
```scala
MaxPooling3D(poolSize = Array(2, 2, 2), strides = null, dimOrdering = "CHANNEL_FIRST", inputShape = null)
```
**Python:**
```python
MaxPooling3D(pool_size=(2, 2, 2), strides=None, border_mode="valid", dim_ordering="th", input_shape=None)
```

Parameters:

* `poolSize`: Int array of length 3. Factors by which to downscale (dim1, dim2, dim3). Default is (2, 2, 2), which will halve the image in each dimension.
* `strides`: Int array of length 3. Stride values. Default is null, and in this case it will be equal to poolSize.
* `dimOrdering`: Format of input data. Please use 'CHANNEL_FIRST' (dimOrdering='th').
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, MaxPooling3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(MaxPooling3D(inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
-0.5052603	0.8938585	0.44785392
-0.48919395	0.35026026	0.541859

(1,1,2,.,.) =
1.5306468	0.24512683	1.71524
-0.49025944	2.1886358	0.15880944

(1,2,1,.,.) =
-0.5133986	-0.16549884	-0.2971134
1.5887301	1.8269571	1.3843931

(1,2,2,.,.) =
0.07515256	1.6993935	-0.3392596
1.2611006	0.20215735	1.3105171

(2,1,1,.,.) =
-2.0070438	0.35554957	0.21326075
-0.4078646	-1.5748956	-1.1007504

(2,1,2,.,.) =
1.0571382	-1.6031493	1.4638771
-0.25891435	1.4923956	-0.24045596

(2,2,1,.,.) =
-0.57790893	0.14577095	1.3165486
0.81937057	-0.3797079	1.2544848

(2,2,2,.,.) =
-0.42183575	-0.63774794	-2.0576336
0.43662143	1.9010457	-0.061519064

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
2.1886358

(1,2,1,.,.) =
1.8269571

(2,1,1,.,.) =
1.4923956

(2,2,1,.,.) =
1.9010457

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x1x1x1]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import MaxPooling3D

model = Sequential()
model.add(MaxPooling3D(input_shape = (2, 2, 3, 4)))
input = np.random.random([2, 2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[[0.84753535 0.50434685 0.52747172 0.33269706]
    [0.34719995 0.56281034 0.84342014 0.60542082]
    [0.0075915  0.5172689  0.61403476 0.93090752]]

   [[0.10790405 0.92850679 0.58076877 0.15428097]
    [0.53095263 0.84994739 0.16819276 0.84422279]
    [0.08839155 0.74912927 0.99761196 0.88315599]]]


  [[[0.30663357 0.86370593 0.91039506 0.13289111]
    [0.0454932  0.77708708 0.13107967 0.93576343]
    [0.36460921 0.1927105  0.60364875 0.73543304]]

   [[0.71262306 0.26078261 0.93590841 0.99436626]
    [0.19388108 0.19500226 0.98440322 0.91417912]
    [0.29945682 0.59392383 0.00325553 0.23172411]]]]



 [[[[0.08088128 0.59952297 0.01342958 0.46435589]
    [0.75531065 0.51514193 0.77715928 0.01082167]
    [0.56199609 0.84202426 0.37361701 0.25513394]]

   [[0.24048523 0.28522588 0.57571503 0.85585273]
    [0.62541179 0.68030543 0.82007978 0.20283455]
    [0.73385951 0.75750966 0.38316951 0.47064059]]]


  [[[0.63921505 0.07319605 0.19223147 0.70518361]
    [0.26917124 0.30180913 0.26639086 0.93485296]
    [0.85613515 0.50717733 0.53115565 0.36661366]]

   [[0.30131816 0.54870789 0.47258878 0.55934124]
    [0.6857046  0.39891689 0.35244649 0.70104095]
    [0.76650326 0.4018181  0.88933498 0.83342255]]]]]
```
Output is:
```python
[[[[[0.9285068  0.8442228 ]]]


  [[[0.86370593 0.9943663 ]]]]



 [[[[0.75531065 0.8558527 ]]]


  [[[0.6857046  0.93485296]]]]]
```

---
## **AveragePooling1D**
Average pooling for temporal data.

The input of this layer should be 3D.

**Scala:**
```scala
AveragePooling1D(poolLength = 2, stride = -1, borderMode = "valid", inputShape = null)
```
**Python:**
```python
AveragePooling1D(pool_length=2, stride=None, border_mode="valid", input_shape=None)
```

Parameters:

* `poolLength`: Size of the region to which average pooling is applied. Integer. Default is 2.
* `stride`: Factor by which to downscale. Positive integer, or -1. 2 will halve the input. If -1, it will default to poolLength. Default is -1, and in this case it will be equal to poolSize.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, AveragePooling1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(AveragePooling1D(poolLength = 3, inputShape = Shape(4, 5)))
val input = Tensor[Float](3, 4, 5).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-1.1377933	 0.21371832	  1.7306958	   -1.9029621	-1.4344455
0.30368176	 1.4264593	  -2.1044374   1.9331307	0.064429775
-0.20389123	 1.0778805	  -0.6283651   1.3097609	-0.13545972
0.2993623	 0.6173592	  0.36244655   0.79175955	0.79752296

(2,.,.) =
-0.2151101	 -0.016314683 0.42787352   0.8266788	1.3463322
-0.5822824	 -0.80566406  1.8474609	   1.0040557	0.058591228
1.1027422	 -1.3031522	  -0.17601672  1.0220417	-0.26774135
0.5274945	 0.33779684	  -0.85662115  0.057247106	-0.26438802

(3,.,.) =
-0.069942534 0.9225811	  -0.46108767  2.4335458	0.101546675
-0.12930758	 0.7706995	  -0.1920893   -0.23971881	0.72432745
0.55851805	 -0.5315623	  0.7103099    -0.5954772	1.1504582
0.6810412	 2.08239	  0.5578813    -0.21148366	0.6381254

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4x5]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.34600094	0.9060194	-0.33403555	0.4466432	-0.50182515

(2,.,.) =
0.10178324	-0.70837694	0.69977254	0.95092535	0.37906072

(3,.,.) =
0.11975598	0.38723943	0.01904432	0.5327832	0.6587774

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x1x5]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import AveragePooling1D

model = Sequential()
model.add(AveragePooling1D(pool_length = 3, input_shape = (4, 5)))
input = np.random.random([3, 4, 5])
output = model.forward(input)
```
Input is:
```python
[[[0.70712635 0.03582033 0.06284402 0.32036469 0.81474437]
  [0.79836044 0.90868531 0.28817162 0.82013972 0.31323463]
  [0.22585456 0.36409411 0.22224669 0.93922795 0.19095179]
  [0.39250119 0.64782355 0.85172164 0.28749378 0.1088145 ]]

 [[0.04373644 0.29722078 0.3117768  0.64487829 0.4810562 ]
  [0.3168246  0.08202731 0.58480522 0.72992227 0.64433289]
  [0.49511033 0.09427843 0.80680702 0.23613413 0.70898751]
  [0.50461138 0.26695611 0.34203601 0.09773049 0.19039967]]

 [[0.75294793 0.55036481 0.26584527 0.98080601 0.43339867]
  [0.50389323 0.07068883 0.78938881 0.96551069 0.15544646]
  [0.12795345 0.23093578 0.22171131 0.54183322 0.39152313]
  [0.53546306 0.66279754 0.52490436 0.14028357 0.40409458]]]
```
Output is:
```python
[[[0.57711375 0.4361999  0.19108744 0.69324416 0.43964362]]

 [[0.2852238  0.15784217 0.56779635 0.53697824 0.61145884]]

 [[0.4615982  0.28399646 0.42564845 0.8293833  0.3267894 ]]]
```

---
## **AveragePooling2D**
Average pooling operation for spatial data.

The input of this layer should be 4D.

**Scala:**
```scala
AveragePooling2D(poolSize = Array(2, 2), strides = null, borderMode = "valid", dimOrdering = DataFormat.NCHW, inputShape = null)
```
**Python:**
```python
AveragePooling2D(pool_size=(2, 2), strides=None, border_mode="valid", dim_ordering="th", input_shape=None)
```

Parameters:

* `poolSize`: Int array of length 2 corresponding to the downscale vertically and horizontally. Default is (2, 2), which will halve the image in each dimension.
* `strides`: Int array of length 2. Stride values. Default is null, and in this case it will be equal to poolSize.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `dimOrdering`: Format of input data. Either DataFormat.NCHW (dimOrdering='th') or DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, AveragePooling2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(AveragePooling2D(inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.9929163	0.73885435	-0.34893242	1.1493853
-0.45246652	-0.32470804	1.2192643	0.30351913
1.3251832	0.52051955	-1.1398637	-2.427732

(1,2,.,.) =
-0.5123787	-0.5055035	0.3858232	0.71986055
0.9580216	0.36081943	1.4867425	0.9852266
-0.6051215	-0.15555465	-1.4472512	0.51882136

(2,1,.,.) =
-1.5209191	0.006158142	1.5162845	-0.06919313
0.56743985	-0.499725	-0.44013703	-0.12666322
0.78009427	1.9432178	1.4082893	-0.6143322

(2,2,.,.) =
-1.387891	0.023748515	-0.8295103	-0.9282333
1.1375008	-1.4631946	-0.67415875	-0.7773346
-2.297338	1.0384767	1.7125391	-1.7680352

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.23864903	0.5808091

(1,2,.,.) =
0.075239725	0.89441323

(2,1,.,.) =
-0.36176154	0.22007278

(2,2,.,.) =
-0.4224591	-0.8023093

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x1x2]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import AveragePooling2D

model = Sequential()
model.add(AveragePooling2D(pool_size=(2, 1), input_shape = (2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.47914521 0.73539951 0.06171461 0.89467653]
   [0.31234563 0.17271756 0.07018152 0.8024166 ]
   [0.06515028 0.44676252 0.4729213  0.61605111]]

  [[0.28739335 0.02873526 0.7195863  0.25426542]
   [0.70731384 0.63144869 0.26336958 0.86892886]
   [0.37042182 0.02806922 0.03134913 0.31307453]]]


 [[[0.03581609 0.90237509 0.18110825 0.53525192]
   [0.88061135 0.91222614 0.70604626 0.2646703 ]
   [0.52454525 0.58735032 0.71977602 0.98559032]]

  [[0.96577742 0.70108055 0.1758514  0.50426252]
   [0.64488368 0.05512992 0.01201226 0.61127524]
   [0.62425433 0.768783   0.70864931 0.12786782]]]]
```
Output is:
```python
[[[[0.3957454  0.45405853 0.06594807 0.84854656]]

  [[0.4973536  0.33009198 0.49147797 0.5615971 ]]]


 [[[0.45821372 0.9073006  0.44357726 0.3999611 ]]

  [[0.8053305  0.37810525 0.09393183 0.5577689 ]]]]
```

---
## **AveragePooling3D**
Applies average pooling operation for 3D data (spatial or spatio-temporal).

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

Border mode currently supported for this layer is 'valid'.

The input of this layer should be 5D.

**Scala:**
```scala
AveragePooling3D(poolSize = Array(2, 2, 2), strides = null, dimOrdering = "CHANNEL_FIRST", inputShape = null)
```
**Python:**
```python
AveragePooling3D(pool_size=(2, 2, 2), strides=None, border_mode="valid", dim_ordering="th", input_shape=None)
```

Parameters:

* `poolSize`: Int array of length 3. Factors by which to downscale (dim1, dim2, dim3). Default is (2, 2, 2), which will halve the image in each dimension.
* `strides`: Int array of length 3. Stride values. Default is null, and in this case it will be equal to poolSize.
* `dimOrdering`: Format of input data. Please use 'CHANNEL_FIRST' (dimOrdering='th').
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, AveragePooling3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(AveragePooling3D(inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
-1.2491689	-0.72333497	0.076971635
-1.2854124	-0.026098572	2.1735003

(1,1,2,.,.) =
1.4519714	1.1690209	2.3558137
0.53576165	0.544173	1.1044264

(1,2,1,.,.) =
-1.2603238	-0.5580594	-0.91401285
-0.18393324	-0.34946147	-0.5833402

(1,2,2,.,.) =
-0.2528762	0.5091298	0.2399745
1.4895978	-1.3734508	-1.0218369

(2,1,1,.,.) =
-1.7266496	-0.04624697	0.47165343
0.16339892	0.9384256	1.0018257

(2,1,2,.,.) =
-0.45763373	0.41072395	0.3123065
-1.1914686	0.90784425	-2.8544335

(2,2,1,.,.) =
0.81638193	-1.2425674	1.9570643
1.444956	0.37828556	-1.7336447

(2,2,2,.,.) =
-0.43858975	0.91795254	0.3359727
0.20638026	-0.07622202	-2.1452882

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
0.052114002

(1,2,1,.,.) =
-0.24742219

(2,1,1,.,.) =
-0.12520081

(2,2,1,.,.) =
0.25082216

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x1x1x1]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import AveragePooling3D

model = Sequential()
model.add(AveragePooling3D(input_shape = (2, 2, 3, 4)))
input = np.random.random([2, 2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[[0.13485756 0.51795663 0.80018765 0.69894299]
    [0.36405734 0.4078976  0.54842186 0.09682389]
    [0.4211948  0.02617479 0.00288834 0.35545069]]

   [[0.50913344 0.02088464 0.23779884 0.52915617]
    [0.26135743 0.63541476 0.98262722 0.38933223]
    [0.06365635 0.71485008 0.9092095  0.00912118]]]


  [[[0.27838368 0.73952601 0.22326478 0.87770435]
    [0.56166616 0.91980417 0.87958524 0.64478151]
    [0.27223399 0.43279067 0.56501981 0.8952789 ]]

   [[0.73747941 0.77723745 0.36951866 0.17479712]
    [0.15545437 0.76033322 0.08025347 0.17189757]
    [0.75895885 0.32075913 0.99673581 0.09631574]]]]



 [[[[0.85973584 0.8686678  0.0627561  0.50474641]
    [0.58258002 0.97293251 0.87511651 0.94003187]
    [0.34547194 0.30969372 0.00280887 0.18566383]]

   [[0.45304482 0.38255272 0.59996713 0.47015079]
    [0.66700015 0.02216949 0.28792252 0.64505342]
    [0.07664707 0.0513936  0.85191806 0.2949295 ]]]


  [[[0.19128504 0.15022612 0.34093811 0.41038156]
    [0.4300465  0.31821395 0.32264938 0.57624452]
    [0.75571906 0.3795846  0.74243055 0.04446947]]

   [[0.74883929 0.49201326 0.6636284  0.56709131]
    [0.30872169 0.95402082 0.67219773 0.42247665]
    [0.6475097  0.21798046 0.4543811  0.34647955]]]]]
```
Output is:
```python
[[[[[0.35644493 0.53541136]]]


  [[[0.61623555 0.42772534]]]]



 [[[[0.6010855  0.5482181 ]]]


  [[[0.44917083 0.49695092]]]]]
```

---
## **GlobalAveragePooling1D**
Global average pooling operation for temporal data.

The input of this layer should be 3D.

**Scala:**
```scala
GlobalAveragePooling1D(inputShape = null)
```
**Python:**
```python
GlobalAveragePooling1D(input_shape=None)
```

Parameters:

* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GlobalAveragePooling1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(GlobalAveragePooling1D(inputShape = Shape(2, 3)))
val input = Tensor[Float](2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.52390736	-0.2733816	0.124149635
-1.351596	-1.1435038	-1.5176618

(2,.,.) =
1.0428048	-0.65227276	-0.44158915
-0.23790422	0.4179904	-0.12358317

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
-0.41384432	-0.7084427	-0.69675606
0.40245032	-0.11714119	-0.28258616
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import GlobalAveragePooling1D

model = Sequential()
model.add(GlobalAveragePooling1D(input_shape = (4, 5)))
input = np.random.random([3, 4, 5])
output = model.forward(input)
```
Input is:
```python
[[[0.41830482 0.51504343 0.16341845 0.32686968 0.45131209]
  [0.65222913 0.2332579  0.30523204 0.66523749 0.1743782 ]
  [0.96500534 0.01471419 0.02973987 0.297514   0.26539896]
  [0.81688177 0.99103135 0.95245507 0.61071527 0.64181819]]

 [[0.67188735 0.66448284 0.98463842 0.70925542 0.79625704]
  [0.43972723 0.78854179 0.29135935 0.67825481 0.148575  ]
  [0.14697146 0.82085139 0.99232289 0.6259919  0.31344956]
  [0.59099227 0.807742   0.88554049 0.50057221 0.72404043]]

 [[0.57586565 0.14940891 0.23417466 0.20219842 0.33200183]
  [0.99391725 0.00967387 0.23496968 0.73669051 0.80370055]
  [0.87012358 0.39090161 0.02632224 0.19713041 0.67099818]
  [0.69929419 0.51659386 0.72596856 0.07191141 0.51766764]]]
```
Output is:
```python
[[0.7131053  0.43851173 0.36271137 0.4750841  0.38322684]
 [0.4623946  0.77040446 0.78846526 0.6285186  0.4955805 ]
 [0.7848002  0.26664457 0.30535877 0.3019827  0.581092  ]]
```

---
## **GlobalAveragePooling2D**
Global average pooling operation for spatial data.

The input of this layer should be 4D.

**Scala:**
```scala
GlobalAveragePooling2D(dimOrdering = DataFormat.NCHW, inputShape = null)
```
**Python:**
```python
GlobalAveragePooling2D(dim_ordering="th", input_shape=None)
```

Parameters:

* `dimOrdering`: Format of input data. Either DataFormat.NCHW (dimOrdering='th') or DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GlobalAveragePooling2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(GlobalAveragePooling2D(inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.110688895	-0.95155084	1.8221924	-2.0326483
-0.013243215	-1.2766567	-0.16704278	-0.97121066
-2.4606674	-0.24223651	-0.5687073	0.69842345

(1,2,.,.) =
0.14165956	0.17032783	2.5329256	0.011501087
-0.3236992	1.1332442	0.18139894	-2.3126595
0.1546373	0.35264283	-0.04404357	-0.70906943

(2,1,.,.) =
-0.08527824	0.29270124	-0.7355773	-0.6026267
-0.71629876	0.83938205	0.5129336	0.118145116
0.17555784	-0.8842884	0.12628363	-0.5556226

(2,2,.,.) =
0.6230317	0.64954233	-1.3002442	-0.44802713
-0.7294096	0.29014868	-0.55649257	2.1427174
0.0146621745	0.67039204	0.12979278	1.8543824

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
-0.5043883	0.10740545
-0.12622404	0.27837467
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import GlobalAveragePooling2D

model = Sequential()
model.add(GlobalAveragePooling2D(input_shape = (2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.44851152 0.94140516 0.45500829 0.07239139]
   [0.58724461 0.7386701  0.69641719 0.70497337]
   [0.15950558 0.56006247 0.82534941 0.59303245]]

  [[0.94628326 0.75747177 0.92495215 0.16233194]
   [0.21553426 0.65968036 0.72130258 0.8929379 ]
   [0.91295078 0.36362834 0.04734189 0.32399088]]]


 [[[0.74069289 0.8804913  0.38783329 0.82279268]
   [0.29561186 0.86405938 0.21608269 0.618583  ]
   [0.16823803 0.65690701 0.85394726 0.94541932]]

  [[0.33876558 0.47517543 0.25908204 0.81933296]
   [0.16176792 0.57166    0.28295922 0.95254489]
   [0.10532106 0.98495855 0.41048516 0.86755462]]]]
```
Output is:
```python
[[0.5652142 0.5773672]
 [0.6208883 0.519134 ]]
```

---
## **GlobalAveragePooling3D**
Applies global average pooling operation for 3D data.

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

Border mode currently supported for this layer is 'valid'.

The input of this layer should be 5D.

**Scala:**
```scala
GlobalAveragePooling3D(dimOrdering = "CHANNEL_FIRST", inputShape = null)
```
**Python:**
```python
GlobalAveragePooling3D(dim_ordering="th", input_shape=None)
```

Parameters:

* `dimOrdering`: Format of input data. Please use 'CHANNEL_FIRST' (dimOrdering='th').
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GlobalAveragePooling3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(GlobalAveragePooling3D(inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
1.8996966	 -0.20018125	-0.3271749
0.27594963	 -1.0520669	0.86003053

(1,1,2,.,.) =
-0.7652662	 0.72945994	 0.9008456
0.8692407	 -1.1327444	 2.0664887

(1,2,1,.,.) =
0.10636215	 -0.812925   -0.3757974
0.48761207	 0.017417012 -2.395701

(1,2,2,.,.) =
-1.3122851	 -0.5942121	 -0.6180062
-0.032230377 -0.27521232 -0.3567782

(2,1,1,.,.) =
1.8668615	 -0.4244298	 1.0701258
0.63794065	 -1.023562	 0.16939393

(2,1,2,.,.) =
0.20582832	 0.5321886	 -1.5412451
-0.38068503	 1.4506307	 -0.47838798

(2,2,1,.,.) =
-0.7344984	 -0.28647164 2.410416
-1.8175911	 -1.1973995	 1.001777

(2,2,2,.,.) =
-0.09646813	 0.11988298	 1.4687495
1.493955	 0.16738588	 1.133337

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.34368983	-0.51347965
0.17372166	0.30525622
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import GlobalAveragePooling3D

model = Sequential()
model.add(GlobalAveragePooling3D(input_shape = (2, 2, 3, 4)))
input = np.random.random([2, 2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[[0.32432335 0.73832213 0.55036152 0.86129318]
    [0.77743469 0.17751485 0.91312554 0.24866223]
    [0.15203726 0.08755524 0.75026344 0.68861154]]

   [[0.13905004 0.17479675 0.60677173 0.721492  ]
    [0.81793893 0.53206943 0.05672886 0.09086074]
    [0.38062788 0.74759024 0.68193411 0.59527354]]]


  [[[0.8279544  0.77162248 0.35870929 0.43733777]
    [0.52811695 0.81754727 0.75937201 0.17814556]
    [0.05580855 0.78653563 0.9703449  0.05411087]]

   [[0.22201094 0.62057351 0.2731341  0.31092695]
    [0.67349717 0.55015705 0.96929428 0.92201811]
    [0.70162709 0.36412647 0.9793207  0.16219098]]]]



 [[[[0.56207549 0.96783676 0.3746153  0.72067359]
    [0.72674531 0.03268906 0.05854376 0.98691757]
    [0.55301432 0.22960094 0.42699025 0.15436922]]

   [[0.99067658 0.05304619 0.03752985 0.20063301]
    [0.72412957 0.78939245 0.13619969 0.05445006]
    [0.98083413 0.39714618 0.77932477 0.85357216]]]


  [[[0.95751583 0.35240109 0.91734527 0.53400394]
    [0.58361827 0.53664137 0.58344827 0.09072018]
    [0.07709009 0.04663983 0.47395611 0.53284686]]

   [[0.85101289 0.82861355 0.53598707 0.61957953]
    [0.21639125 0.72122269 0.86056859 0.52517511]
    [0.8124116  0.60072907 0.49943728 0.71011677]]]]]
```
Output is:
```python
[[0.49227658 0.5539368 ]
 [0.4912919  0.5611447 ]]
```

---
## **GlobalMaxPooling1D**
Global max pooling operation for temporal data.

The input of this layer should be 3D.

**Scala:**
```scala
GlobalMaxPooling1D(inputShape = null)
```
**Python:**
```python
GlobalMaxPooling1D(input_shape=None)
```

Parameters:

* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GlobalMaxPooling1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(GlobalMaxPooling1D(inputShape = Shape(2, 3)))
val input = Tensor[Float](2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-1.0603509	2.0034506	-1.9884554
-1.187076	-1.3023934	-0.8058352

(2,.,.) =
-0.9960039	-2.5800185	-0.01848254
-0.66063184	-1.451372	1.3490999

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
-1.0603509	2.0034506	-0.8058352
-0.66063184	-1.451372	1.3490999
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import GlobalMaxPooling1D

model = Sequential()
model.add(GlobalMaxPooling1D(input_shape = (2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.70782546 0.2064717  0.45753652]
  [0.41262607 0.72777349 0.21662695]]

 [[0.0937254  0.16749913 0.65395922]
  [0.51027108 0.67591602 0.41025529]]]
```
Output is:
```python
[[0.7078255  0.7277735  0.45753652]
 [0.5102711  0.675916   0.6539592 ]]
```

---
## **GlobalMaxPooling2D**
Global max pooling operation for spatial data.

The input of this layer should be 4D.

**Scala:**
```scala
GlobalMaxPooling2D(dimOrdering = DataFormat.NCHW, inputShape = null)
```
**Python:**
```python
GlobalMaxPooling2D(dim_ordering="th", input_shape=None)
```

Parameters:

* `dimOrdering`: Format of input data. Either DataFormat.NCHW (dimOrdering='th') or DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GlobalMaxPooling2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(GlobalMaxPooling2D(inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-0.33040258	-0.94677037	0.14672963	 0.1252591
-0.3141728	0.68040586	1.3511285	 -0.29235613
0.03947779	-1.1260709	0.16128083	 -1.1656744

(1,2,.,.) =
1.0182372	-0.6030568	-1.5335841	 0.37804475
0.26944965	-1.6720067	0.2405665	 -0.95661074
-0.31286374	0.109459646	-1.6644431	 -1.9295278

(2,1,.,.) =
1.0210015	-0.69647574	-0.629564	 1.6719679
-0.7825565	-0.48921636	0.1892077	 0.17827414
0.76913565	0.17354056	-0.5749589	 -1.736962

(2,2,.,.) =
0.82071537	-0.22566034	0.12415939	 0.02941268
0.34600595	0.86877316	0.9797952	 -1.7793267
0.025843443	-1.6373945	-0.093925744 -0.22479358

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
1.3511285	1.0182372
1.6719679	0.9797952
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import GlobalMaxPooling2D

model = Sequential()
model.add(GlobalMaxPooling2D(input_shape = (2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.4037502  0.02214535 0.83987632 0.63656679]
   [0.57757778 0.41884406 0.94446159 0.55776242]
   [0.40213234 0.4463349  0.04457756 0.99123233]]

  [[0.71114675 0.88609155 0.40299526 0.01648121]
   [0.23102758 0.89673552 0.07030409 0.79674961]
   [0.84665248 0.18257089 0.87211872 0.22697933]]]


 [[[0.08033122 0.26298654 0.10863184 0.57894922]
   [0.03999134 0.90867755 0.80473921 0.79913378]
   [0.60443084 0.92055786 0.17994007 0.87414516]]

  [[0.50193442 0.52639178 0.72124789 0.41776979]
   [0.09495006 0.91797563 0.48755794 0.50458372]
   [0.47387433 0.93445126 0.83216554 0.67275364]]]]
```
Output is:
```python
[[0.99123234 0.8967355 ]
 [0.92055786 0.9344513 ]]
```

---
## **GlobalMaxPooling3D**
Applies global max pooling operation for 3D data.

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

Border mode currently supported for this layer is 'valid'.

The input of this layer should be 5D.

**Scala:**
```scala
GlobalMaxPooling3D(dimOrdering = "CHANNEL_FIRST", inputShape = null)
```
**Python:**
```python
GlobalMaxPooling3D(dim_ordering="th", input_shape=None)
```

Parameters:

* `dimOrdering`: Format of input data. Please use 'CHANNEL_FIRST' (dimOrdering='th').
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GlobalMaxPooling3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(GlobalMaxPooling3D(inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
0.5882856	0.6403546	-0.42084476
-0.097307995	-0.52530056	-0.36757985

(1,1,2,.,.) =
0.60983604	-0.24730012	-0.07744695
0.36276138	0.34722528	0.19874145

(1,2,1,.,.) =
-0.912143	0.14050196	2.1116483
-0.67692965	-0.5708391	-2.1040971

(1,2,2,.,.) =
2.1500692	1.1697202	1.364164
1.2241726	-0.12069768	1.2471954

(2,1,1,.,.) =
0.39550102	-0.7435119	0.47669584
-0.17335615	0.2690476	-0.8462402

(2,1,2,.,.) =
-1.0553921	-0.35153934	0.8036665
-1.029019	-0.64534503	0.94537926

(2,2,1,.,.) =
0.5388452	-0.27233714	1.5837694
1.0976856	-0.20959699	1.6285672

(2,2,2,.,.) =
-0.7736055	0.58593166	-1.2158531
1.2194971	1.4081163	1.2056179

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.6403546	2.1500692
0.94537926	1.6285672
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import GlobalMaxPooling3D

model = Sequential()
model.add(GlobalMaxPooling3D(input_shape = (2, 2, 3, 4)))
input = np.random.random([2, 2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[[0.42034371 0.63078488 0.22453127 0.16344976]
    [0.36893331 0.33043678 0.39829539 0.18374732]
    [0.81114306 0.63929787 0.89367189 0.55747888]]

   [[0.91527128 0.30714881 0.37284006 0.72261292]
    [0.70570501 0.34112634 0.40325762 0.13267086]
    [0.00591885 0.34739306 0.18708168 0.20429708]]]


  [[[0.4275678  0.4519255  0.79702921 0.1790105 ]
    [0.33051034 0.3862867  0.31801831 0.69477123]
    [0.64636877 0.01875691 0.5253049  0.1629054 ]]

   [[0.08784506 0.07041528 0.45395457 0.64341618]
    [0.24214838 0.56265812 0.5308968  0.91477167]
    [0.56046541 0.1611661  0.59909403 0.05736846]]]]



 [[[[0.14884228 0.27530387 0.20530849 0.22477436]
    [0.05815231 0.4457146  0.13238115 0.73269285]
    [0.72159769 0.97602167 0.52434989 0.45926641]]

   [[0.72613216 0.41068657 0.91639059 0.44668906]
    [0.51405858 0.48933665 0.96326869 0.1527533 ]
    [0.56420469 0.95666582 0.14072945 0.26675841]]]


  [[[0.65052301 0.23831108 0.90755926 0.72338704]
    [0.06441496 0.20477211 0.82261371 0.41171546]
    [0.56304926 0.65914837 0.66960326 0.27941308]]

   [[0.35223348 0.87843857 0.88300181 0.73362138]
    [0.24517247 0.26258726 0.5723223  0.92380302]
    [0.9226784  0.6437559  0.45734009 0.28308591]]]]]
```
Output is:
```python
[[0.9152713  0.9147717 ]
 [0.97602165 0.92380303]]
```

---
