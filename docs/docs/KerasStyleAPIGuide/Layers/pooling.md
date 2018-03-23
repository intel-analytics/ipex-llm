## **MaxPooling1D**
Max pooling operation for temporal data.

The input of this layer should be 3D.

**Scala:**
```scala
MaxPooling1D(poolLength = 2, stride = -1, borderMode = "valid", inputShape = null)
```
**Python:**
```python
MaxPooling1D(pool_length=2, stride=None, border_mode="valid", input_shape=None, name=None)
```

Parameters:

* `poolLength`: Size of the region to which max pooling is applied. Integer. Default is 2.
* `stride`: Factor by which to downscale. 2 will halve the input. If not specified, it will default to poolLength.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, MaxPooling1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
MaxPooling2D(poolSize = (2, 2), strides = null, borderMode = "valid", dimOrdering = "th", inputShape = null)
```
**Python:**
```python
MaxPooling2D(pool_size=(2, 2), strides=None, border_mode="valid", dim_ordering="th", input_shape=None, name=None)
```

Parameters:

* `poolSize`: Length 2 corresponding to the downscale vertically and horizontally. Default is (2, 2), which will halve the image in each dimension.
* `strides`: Length 2. Stride values. Default is null, and in this case it will be equal to poolSize.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, MaxPooling2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
model.add(MaxPooling2D(input_shape = (2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.82279705 0.62487892 0.37391352 0.22834848]
   [0.68709158 0.40902972 0.73191486 0.40095294]
   [0.651977   0.93330601 0.45785981 0.45939351]]
  [[0.372833   0.39871945 0.13426243 0.83083849]
   [0.24290548 0.04446027 0.58070741 0.37752852]
   [0.13116942 0.59339663 0.94669915 0.02460278]]]

 [[[0.46505904 0.96103464 0.75846419 0.77357123]
   [0.37835688 0.88438048 0.5679742  0.74607276]
   [0.41415466 0.73945737 0.39188398 0.52736799]]
  [[0.51772064 0.19857965 0.15476197 0.64569767]
   [0.21794751 0.74455093 0.48423447 0.15482331]
   [0.38363071 0.78733222 0.2542284  0.88671892]]]]
```
Output is:
```python
[[[[0.82279706 0.7319149 ]]
  [[0.39871946 0.8308385 ]]]

 [[[0.96103466 0.77357125]]
  [[0.74455094 0.64569765]]]]
```

---
## **MaxPooling3D**
Applies max pooling operation for 3D data (spatial or spatio-temporal).

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

Border mode currently supported for this layer is 'valid'.

The input of this layer should be 5D.

**Scala:**
```scala
MaxPooling3D(poolSize = (2, 2, 2), strides = null, dimOrdering = "th", inputShape = null)
```
**Python:**
```python
MaxPooling3D(pool_size=(2, 2, 2), strides=None, border_mode="valid", dim_ordering="th", input_shape=None, name=None)
```

Parameters:

* `poolSize`: Length 3. Factors by which to downscale (dim1, dim2, dim3). Default is (2, 2, 2), which will halve the image in each dimension.
* `strides`: Length 3. Stride values. Default is null, and in this case it will be equal to poolSize.
* `dimOrdering`: Format of input data. Only 'th' (Channel First) is supported for now.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, MaxPooling3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
model.add(MaxPooling3D(input_shape = (2, 2, 2, 3)))
input = np.random.random([2, 2, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[[[0.73349746 0.9811588  0.86071417]
    [0.33287621 0.37991739 0.87029317]]
   [[0.62537904 0.48099174 0.06194759]
    [0.38747972 0.05175308 0.36096032]]]
  [[[0.63260385 0.69990236 0.63353249]
    [0.19081261 0.56210617 0.75985185]]
   [[0.8624058  0.47224318 0.26524027]
    [0.75317792 0.39251436 0.98938982]]]]

 [[[[0.00556086 0.18833728 0.80340438]
    [0.9317538  0.88142596 0.90724509]]
   [[0.90243612 0.04594116 0.43662143]
    [0.24205094 0.58687822 0.57977055]]]
  [[[0.17240398 0.18346483 0.02520754]
    [0.06968248 0.02442692 0.56078895]]
   [[0.69503427 0.09528588 0.46104647]
    [0.16752596 0.88175901 0.71032998]]]]]
```
Output is:
```python
[[[[[0.9811588]]]
  [[[0.8624058]]]]

 [[[[0.9317538]]]
  [[[0.881759 ]]]]]
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
AveragePooling1D(pool_length=2, stride=None, border_mode="valid", input_shape=None, name=None)
```

Parameters:

* `poolLength`: Size of the region to which average pooling is applied. Integer. Default is 2.
* `stride`: Factor by which to downscale. 2 will halve the input. If not specified, it will default to poolLength.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, AveragePooling1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
AveragePooling2D(poolSize = (2, 2), strides = null, borderMode = "valid", dimOrdering = "th", inputShape = null)
```
**Python:**
```python
AveragePooling2D(pool_size=(2, 2), strides=None, border_mode="valid", dim_ordering="th", input_shape=None, name=None)
```

Parameters:

* `poolSize`: Length 2 corresponding to the downscale vertically and horizontally. Default is (2, 2), which will halve the image in each dimension.
* `strides`: Length 2. Stride values. Default is null, and in this case it will be equal to poolSize.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, AveragePooling2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
model.add(AveragePooling2D(input_shape = (2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.23128514 0.69922098 0.52158685 0.43063779]
   [0.89149649 0.33910949 0.4402748  0.08933058]
   [0.71712488 0.21574851 0.76768248 0.57027882]]
  [[0.08349921 0.85318742 0.49922456 0.6256355 ]
   [0.22331336 0.78402155 0.91424506 0.18895412]
   [0.89722286 0.31067545 0.82655572 0.37775551]]]

 [[[0.9706926  0.28398186 0.36623623 0.23701637]
   [0.49936358 0.50951663 0.48116156 0.89941571]
   [0.06519683 0.34624179 0.2462403  0.48512833]]
  [[0.58408752 0.68318898 0.67886418 0.43403476]
   [0.87328453 0.8412756  0.59168164 0.49972216]
   [0.82188585 0.63685579 0.50966912 0.51439279]]]]
```
Output is:
```python
[[[[0.540278   0.3704575 ]]
  [[0.48600537 0.5570148 ]]]

 [[[0.56588864 0.49595746]]
  [[0.7454592  0.5510757 ]]]]
```

---
## **AveragePooling3D**
Applies average pooling operation for 3D data (spatial or spatio-temporal).

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

Border mode currently supported for this layer is 'valid'.

The input of this layer should be 5D.

**Scala:**
```scala
AveragePooling3D(poolSize = (2, 2, 2), strides = null, dimOrdering = "th", inputShape = null)
```
**Python:**
```python
AveragePooling3D(pool_size=(2, 2, 2), strides=None, border_mode="valid", dim_ordering="th", input_shape=None, name=None)
```

Parameters:

* `poolSize`: Length 3. Factors by which to downscale (dim1, dim2, dim3). Default is (2, 2, 2), which will halve the image in each dimension.
* `strides`: Length 3. Stride values. Default is null, and in this case it will be equal to poolSize.
* `dimOrdering`: Format of input data. Only 'th' (Channel First) is supported for now.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, AveragePooling3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
model.add(AveragePooling3D(input_shape = (2, 2, 2, 3)))
input = np.random.random([2, 2, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[[[0.97536696 0.01017816 0.77256405]
    [0.53928594 0.80710453 0.71903394]]
   [[0.67067647 0.38680811 0.05431165]
    [0.98320357 0.8602754  0.12535218]]]
  [[[0.32317928 0.17129988 0.51584225]
    [0.70060648 0.36568169 0.36353108]]
   [[0.90675921 0.68967216 0.29854921]
    [0.72568459 0.34304905 0.9501725 ]]]]

 [[[[0.96295459 0.51457555 0.15752579]
    [0.29569757 0.73166152 0.24217882]]
   [[0.69938844 0.98315048 0.36022304]
    [0.97079866 0.03950786 0.18505114]]]
  [[[0.10255992 0.87988966 0.13163776]
    [0.286857   0.56472867 0.73914834]]
   [[0.51970598 0.19869426 0.47845175]
    [0.86776147 0.60381965 0.88064078]]]]]
```
Output is:
```python
[[[[[0.6541124 ]]]
  [[[0.5282415 ]]]]

 [[[[0.64971685]]]
  [[[0.5030021 ]]]]]

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
GlobalMaxPooling1D(input_shape=None, name=None)
```

Parameters:

* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GlobalMaxPooling1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
GlobalMaxPooling2D(dimOrdering = "th", inputShape = null)
```
**Python:**
```python
GlobalMaxPooling2D(dim_ordering="th", input_shape=None, name=None)
```

Parameters:

* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GlobalMaxPooling2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
[[0.99123234 0.8967355]
 [0.92055786 0.9344513]]
```

---
## **GlobalMaxPooling3D**
Applies global max pooling operation for 3D data.

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

Border mode currently supported for this layer is 'valid'.

The input of this layer should be 5D.

**Scala:**
```scala
GlobalMaxPooling3D(dimOrdering = "th", inputShape = null)
```
**Python:**
```python
GlobalMaxPooling3D(dim_ordering="th", input_shape=None, name=None)
```

Parameters:

* `dimOrdering`: Format of input data. Only 'th' (Channel First) is supported for now.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GlobalMaxPooling3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GlobalMaxPooling3D(inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
0.5882856	   0.6403546	-0.42084476
-0.097307995   -0.52530056	-0.36757985

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
model.add(GlobalMaxPooling3D(input_shape = (2, 2, 2, 3)))
input = np.random.random([2, 2, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[[[0.35670186 0.12860978 0.99958102]
    [0.17345679 0.19473725 0.41235665]]
   [[0.98948429 0.26686797 0.4632353 ]
    [0.15791828 0.07452679 0.73215605]]]
  [[[0.80305568 0.73208753 0.31179214]
    [0.43452576 0.563038   0.65955869]]
   [[0.31947806 0.00899334 0.55208827]
    [0.57471665 0.10157217 0.42698318]]]]

 [[[[0.59277903 0.35379325 0.5311834 ]
    [0.91781414 0.10407255 0.58049721]]
   [[0.14371521 0.24279466 0.26071055]
    [0.89431752 0.66817043 0.61662462]]]
  [[[0.6672706  0.38855847 0.88462881]
    [0.38859986 0.80439572 0.27661295]]
   [[0.41453042 0.11527795 0.75953012]
    [0.77940987 0.26283438 0.97745039]]]]]
```
Output is:
```python
[[0.99958104 0.8030557]
 [0.91781414 0.9774504]]
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
GlobalAveragePooling1D(input_shape=None, name=None)
```

Parameters:

* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GlobalAveragePooling1D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

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
model.add(GlobalAveragePooling1D(input_shape = (2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.93869359 0.22245741 0.9744004 ]
  [0.89151128 0.8211663  0.73579694]]

 [[0.37929716 0.509159   0.21713254]
  [0.81838451 0.72323228 0.0370643 ]]]
```
Output is:
```python
[[0.9151024   0.52181184 0.85509866]
 [0.59884083  0.6161956  0.12709841]]
```

---
## **GlobalAveragePooling2D**
Global average pooling operation for spatial data.

The input of this layer should be 4D.

**Scala:**
```scala
GlobalAveragePooling2D(dimOrdering = "th", inputShape = null)
```
**Python:**
```python
GlobalAveragePooling2D(dim_ordering="th", input_shape=None, name=None)
```

Parameters:

* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GlobalAveragePooling2D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GlobalAveragePooling2D(inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.110688895	   -0.95155084	1.8221924	-2.0326483
-0.013243215   -1.2766567	-0.16704278	-0.97121066
-2.4606674	   -0.24223651	-0.5687073	0.69842345

(1,2,.,.) =
0.14165956	   0.17032783	2.5329256	0.011501087
-0.3236992	   1.1332442	0.18139894	-2.3126595
0.1546373	   0.35264283	-0.04404357	-0.70906943

(2,1,.,.) =
-0.08527824	   0.29270124	-0.7355773	-0.6026267
-0.71629876	   0.83938205	0.5129336	0.118145116
0.17555784	   -0.8842884	0.12628363	-0.5556226

(2,2,.,.) =
0.6230317	   0.64954233	-1.3002442	-0.44802713
-0.7294096	   0.29014868	-0.55649257	2.1427174
0.0146621745   0.67039204	0.12979278	1.8543824

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
GlobalAveragePooling3D(dimOrdering = "th", inputShape = null)
```
**Python:**
```python
GlobalAveragePooling3D(dim_ordering="th", input_shape=None, name=None)
```

Parameters:

* `dimOrdering`: Format of input data. Only 'th' (Channel First) is supported for now.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, GlobalAveragePooling3D}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GlobalAveragePooling3D(inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
1.8996966	 -0.20018125  -0.3271749
0.27594963	 -1.0520669	  0.86003053

(1,1,2,.,.) =
-0.7652662	 0.72945994	  0.9008456
0.8692407	 -1.1327444	  2.0664887

(1,2,1,.,.) =
0.10636215	 -0.812925    -0.3757974
0.48761207	 0.017417012  -2.395701

(1,2,2,.,.) =
-1.3122851	 -0.5942121	  -0.6180062
-0.032230377 -0.27521232  -0.3567782

(2,1,1,.,.) =
1.8668615	 -0.4244298	  1.0701258
0.63794065	 -1.023562	  0.16939393

(2,1,2,.,.) =
0.20582832	 0.5321886	  -1.5412451
-0.38068503	 1.4506307	  -0.47838798

(2,2,1,.,.) =
-0.7344984	 -0.28647164  2.410416
-1.8175911	 -1.1973995	  1.001777

(2,2,2,.,.) =
-0.09646813	 0.11988298	  1.4687495
1.493955	 0.16738588	  1.133337

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
model.add(GlobalAveragePooling3D(input_shape = (2, 2, 2, 3)))
input = np.random.random([2, 2, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[[[0.38403874 0.30696173 0.25682854]
    [0.53124253 0.62668969 0.21927777]]
   [[0.33040063 0.37388563 0.75210039]
    [0.08358634 0.80063745 0.13251887]]]
  [[[0.41724617 0.2241106  0.55527267]
    [0.69493785 0.71098284 0.54058444]]
   [[0.4773658  0.92236993 0.76933649]
    [0.45217032 0.61153948 0.01976393]]]]

 [[[[0.27256789 0.56008397 0.19898919]
    [0.44973465 0.66605998 0.77117999]]
   [[0.07868799 0.94786045 0.2240451 ]
    [0.92261946 0.4053334  0.2572511 ]]]
  [[[0.33754374 0.28838802 0.79900278]
    [0.26374789 0.25610211 0.9320699 ]]
   [[0.19518511 0.80707822 0.29660536]
    [0.56917623 0.07653736 0.77836375]]]]]
```
Output is:
```python
[[0.3998474  0.53297335]
 [0.47953442 0.46665   ]]
```
