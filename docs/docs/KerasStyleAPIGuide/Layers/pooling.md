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
import com.intel.analytics.zoo.pipeline.api.keras.layers.MaxPooling3D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
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
from zoo.pipeline.api.keras.layers import MaxPooling3D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(MaxPooling3D(input_shape=(2, 2, 2, 3)))
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
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.GlobalMaxPooling2D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GlobalMaxPooling2D[Float](inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.12648843      0.15536028      1.3401515       -0.25693455
0.6002777       0.6886729       -1.0650102      -0.22140503
-0.7598008      0.8800106       -0.061039474    -1.3625065

(1,2,.,.) =
-0.37492484     -0.6727478      -0.12211597     1.3243467
-0.72237        0.6942101       -1.455304       -0.23814173
-0.38509718     -0.9179013      -0.99926376     0.18432678

(2,1,.,.) =
0.4457857       -0.36717635     -0.6653158      -1.9075912
-0.49489713     -0.70543754     0.85306334      0.21031244
0.08930698      0.046588574     0.9523686       -0.87959886

(2,2,.,.) =
-0.8523849      0.55808693      -1.5779148      1.312412
-0.9923541      -0.562809       1.1512411       0.33178216
1.056546        -2.0607772      -0.8233232      0.024466092

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
1.3401515       1.3243467
0.9523686       1.312412
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import GlobalMaxPooling2D

model = Sequential()
model.add(GlobalMaxPooling2D(input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.82189558 0.20668687 0.84669433 0.58740261]
-   [0.33005685 0.93836385 0.51005935 0.11894048]
-   [0.39757919 0.17126568 0.38237808 0.35911186]]
-
-  [[0.98544456 0.10949685 0.47642379 0.21039236]
-   [0.51058537 0.9625007  0.2519618  0.03186033]
-   [0.28042435 0.08481816 0.37535567 0.60848855]]]
-
-
- [[[0.34468892 0.48365864 0.01397789 0.16565704]
-   [0.91387839 0.78507728 0.0912983  0.06167101]
-   [0.49026863 0.17870698 0.43566122 0.79984653]]
-
-  [[0.15157888 0.07546447 0.47063241 0.46052913]
-   [0.92483801 0.51271677 0.45300461 0.40369727]
-   [0.94152848 0.61306339 0.43241425 0.88775481]]]]
```
Output is:
```python
[[0.93836385 0.98544455]
- [0.9138784  0.9415285 ]]
```

---
## **GlobalMaxPooling3D**
Applies global max pooling operation for 3D data.

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

Border mode currently supported for this layer is 'valid'.

The input of this layer should be 5D, i.e. (batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels).
The output of this layer should be 2D, i.e. (batch_size, channels).

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
* `name`: String to set the name of the layer. If not specified, its name will by default to be a generated string.

**Scala example:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.GlobalMaxPooling3D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GlobalMaxPooling3D[Float](inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
0.50938565	1.6374807	0.8158744
-0.3293317	-0.17766304	0.9067782

(1,1,2,.,.) =
1.5450556	-1.0339675	0.056255028
-0.8867852	-0.05401365	-0.9615863

(1,2,1,.,.) =
-0.98946816	0.21851462	-0.4431965
-0.7591889	1.1842074	0.98533714

(1,2,2,.,.) =
-0.12944926	0.58315176	-1.5754528
-0.93392104	-0.38259965	0.3566876

(2,1,1,.,.) =
-0.1219873	-0.06568	0.5519306
0.32932717	1.4409258	0.68309426

(2,1,2,.,.) =
-1.4289209	0.47897565	-1.0722001
-0.64675856	0.7097152	0.31949154

(2,2,1,.,.) =
-0.89986056	-0.13643691	0.69211197
0.08849494	0.8695818	1.5527223

(2,2,2,.,.) =
1.3823601	0.36978078	0.10262361
0.05734055	-0.41569084	0.009035309

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
1.6374807	1.1842074
1.4409258	1.5527223
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import GlobalMaxPooling3D

model = Sequential()
model.add(GlobalMaxPooling3D(input_shape = (2, 2, 2, 3)))
input = np.random.random([2, 2, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
      [[[[[ 0.8402289  0.11503692 0.27831015]
          [ 0.45756199 0.15043262 0.78778086]]

         [[ 0.37076324 0.65032926 0.74508221]
          [ 0.32223229 0.81980455 0.14822856]]]


        [[[ 0.72858223 0.04609062 0.86802821]
          [ 0.22619071 0.23091766 0.68856216]]

         [[ 0.54321111 0.94913088 0.59588331]
          [ 0.90821291 0.42860528 0.39355229]]]]



       [[[[ 0.06834657 0.41250882 0.55612858]
          [ 0.72871084 0.59139003 0.83317638]]

         [[ 0.99382906 0.24782635 0.27295274]
          [ 0.65663701 0.7994264  0.73672449]]]


        [[[ 0.11487664 0.74224294 0.39289158]
          [ 0.34253228 0.47903629 0.66238715]]

         [[ 0.13219379 0.12541975 0.93002441]
          [ 0.58895306 0.38519765 0.27216034]]]]]
```
Output is:
```python
[[ 0.84022892  0.94913089]
 [ 0.99382907  0.93002439]]
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
import com.intel.analytics.zoo.pipeline.api.keras.layers.AveragePooling2D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(AveragePooling2D[Float](inputShape = Shape(2, 3, 4)))
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
from zoo.pipeline.api.keras.layers import AveragePooling2D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(AveragePooling2D(input_shape=(2, 3, 4)))
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
## **AveragePooling1D**
Applies average pooling operation for temporal data.

The input of this layer should be 3D.

**Scala:**
```scala
AveragePooling1D(poolSize = 2, strides = -1, dimOrdering = "valid", inputShape = null)
```
**Python:**
```python
AveragePooling1D(pool_length=2, stride=None, border_mode="valid", input_shape=None, name=None)
```

**Parameters:**

* `poolLength`: Size of the region to which average pooling is applied. Integer. Default is 2.
* `stride`: Factor by which to downscale. Positive integer, or -1. 2 will halve the input.
            If -1, it will default to poolLength. Default is -1, and in this case it will
            be equal to poolSize.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.AveragePooling1D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(AveragePooling1D[Float](inputShape = Shape(2, 3)))
val input = Tensor[Float](2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
2.0454981       -0.9984553      -0.22548687
-2.9674191      0.61953986      0.9267055

(2,.,.) =
0.2458116       -0.06563047     0.11032024
0.29159164      1.0789983       0.6236742

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.4609605      -0.18945771     0.3506093

(2,.,.) =
0.2687016       0.50668395      0.36699724

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x1x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import AveragePooling1D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(AveragePooling1D(input_shape = (2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
array([[[0.27910133, 0.62511864, 0.11819567],
        [0.60144333, 0.17082084, 0.32399398]],

       [[0.44947572, 0.97199261, 0.95654852],
        [0.72464095, 0.50742734, 0.09491157]]])
```
Output is:
```python
array([[[0.44027233, 0.39796975, 0.22109482]],

       [[0.5870583 , 0.73971   , 0.52573   ]]], dtype=float32)
```

---
## **GlobalAveragePooling2D**
Applies global average pooling operation for spatial data.

The input of this layer should be 4D.

**Scala:**
```scala
GlobalAveragePooling2D(dimOrdering = "th", inputShape = null)
```
**Python:**
```python
GlobalAveragePooling2D(dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `dimOrdering`: Format of input data. Please use DataFormat.NCHW (dimOrdering='th') or DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.GlobalAveragePooling2D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GlobalAveragePooling2D[Float](inputShape = Shape(2, 3, 3)))
val input = Tensor[Float](2, 2, 3, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-1.3950379      0.23557353      -1.8424573
0.07449951      0.6322816       0.8831866
0.8229907       1.5395391       -0.84414214

(1,2,.,.) =
2.1792102       -1.0315448      -1.1207858
-1.1498563      1.876386        -0.67528623
0.54306036      0.7579748       0.09953801

(2,1,.,.) =
-0.5101911      -1.1826278      -0.5852779
0.53600776      0.6960143       -2.8790317
-0.4959711      -1.2831435      -0.09703717

(2,2,.,.) =
0.5213661       -0.4794566      -0.48301712
0.3673898       -0.048692267    -0.043640807
-0.60638505     -0.07805356     1.2334769

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.011825972     0.16429958
-0.64458424     0.04255416
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import GlobalAveragePooling2D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(GlobalAveragePooling2D(input_shape = (2, 3, 3)))
input = np.random.random([2, 2, 3, 3])
output = model.forward(input)
```
Input is:
```python
array([[[[0.54771885, 0.53283909, 0.46927443],
         [0.47621227, 0.76883995, 0.52303474],
         [0.60008681, 0.60752329, 0.98198994]],

        [[0.28667601, 0.47522264, 0.4943029 ],
         [0.00561534, 0.39171735, 0.23420212],
         [0.50868123, 0.40796681, 0.82682555]]],
       [[[0.78836132, 0.58607316, 0.93814738],
         [0.34578363, 0.32976447, 0.49251034],
         [0.22992651, 0.04771577, 0.56071013]],

        [[0.34291469, 0.13181605, 0.68202722],
         [0.16404025, 0.54052442, 0.79312374],
         [0.0254005 , 0.71477398, 0.94485338]]]])
```
Output is:
```python
array([[0.61194664, 0.40346777],
       [0.47988808, 0.4821638 ]], dtype=float32)
```