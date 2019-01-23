## **MaxPooling1D**
Max pooling operation for temporal data.

The input is 3D tensor with shape:(batch_size, steps, feature_dim).

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
import com.intel.analytics.zoo.pipeline.api.keras.layers.MaxPooling1D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential

val model = Sequential[Float]()
model.add(MaxPooling1D[Float](poolLength = 3, inputShape = Shape(4, 5)))
val input = Tensor[Float](3, 4, 5).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-1.2339195      -1.2134796      0.16991705      -0.10169973     -1.2464932
0.37946555      0.29533234      -1.2552645      -2.6928735      -0.44519955
0.98743796      -1.0912303      -0.13897413     1.0241779       -0.5951304
-0.31459442     -0.088579334    -0.58336115     -0.6427486      -0.1447043

(2,.,.) =
0.14750746      0.07493488      -0.8554524      -1.6551514      0.16679412
-0.82279974     0.25704315      0.09921734      -0.8135057      2.7640774
-1.0111052      0.34388593      -0.7569789      1.0547938       1.6738676
0.4396624       -1.0570261      0.061429325     1.1752373       -0.14648575

(3,.,.) =
-0.95818335     0.8790822       -0.99111855     -0.9717616      -0.39238095
1.2533073       0.23365906      1.7784269       1.0600376       1.6816885
0.7145845       0.4711851       -0.4465603      -0.77884597     0.484986
0.42429695      -2.00715        0.6520644       1.3022201       -0.48169184


[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4x5]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.98743796      0.29533234      0.16991705      1.0241779       -0.44519955

(2,.,.) =
0.14750746      0.34388593      0.09921734      1.0547938       2.7640774

(3,.,.) =
1.2533073       0.8790822       1.7784269       1.0600376       1.6816885

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x1x5]

```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import MaxPooling1D

model = Sequential()
model.add(MaxPooling1D(pool_length = 3, input_shape = (4, 5)))
input = np.random.random([3, 4, 5])
output = model.forward(input)
```
Input is:
```python
[[[0.14508341 0.42297648 0.50516337 0.15659868 0.83121192]
  [0.27837702 0.87282932 0.94292864 0.48428998 0.23604637]
  [0.24147633 0.2116796  0.54433489 0.22961905 0.88685975]
  [0.57235359 0.16278372 0.39749189 0.20781401 0.22834635]]

 [[0.42306184 0.43404804 0.22141668 0.0316458  0.08445576]
  [0.88377164 0.00417697 0.52975728 0.43238725 0.40539813]
  [0.90702837 0.37940347 0.06435512 0.33566794 0.50049895]
  [0.12146178 0.61599986 0.11874934 0.57207512 0.87713768]]

 [[0.56690324 0.99869154 0.87789702 0.67840158 0.64935853]
  [0.9950283  0.55710408 0.70919634 0.52309929 0.14311439]
  [0.25394468 0.41519219 0.8074057  0.05341861 0.98447171]
  [0.71387206 0.74763239 0.27057394 0.09578605 0.68601852]]]
```
Output is:
```python
[[[0.27837703 0.8728293  0.9429287  0.48428997 0.8868598 ]]

 [[0.9070284  0.43404803 0.52975726 0.43238723 0.50049895]]

 [[0.9950283  0.99869156 0.877897   0.6784016  0.98447174]]]
```

---
## **MaxPooling2D**
Max pooling operation for spatial data.

The input is 4D tensor with shape:(batch_size, rows, cols, channels).
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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.MaxPooling2D

val model = Sequential[Float]()
model.add(MaxPooling2D[Float](inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.02138003      -0.20666665     -0.93250555     0.41267508
-0.40883347     0.4919021       0.7189889       1.3442185
-0.08697278     -0.025719838    2.1126  0.69069535

(1,2,.,.) =
-0.1685801      -0.07843445     1.3499486       -0.5944459
0.29377022      0.061167963     -0.60608864     -0.08283464
0.03402891      -1.0627178      1.9463096       0.0011169242

(2,1,.,.) =
-1.4524128      1.3868454       2.3057284       1.574949
-1.165581       0.79445213      -0.63500565     -0.17981622
-0.98042095     -1.7876958      0.8024988       -0.90554804

(2,2,.,.) =
-1.6468426      1.1864686       -0.683854       -1.5643677
2.8272789       -0.5537863      -0.563258       -0.01623243
-0.31333938     0.03472893      -1.730748       -0.15463233

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output:
(1,1,.,.) =
0.4919021       1.3442185

(1,2,.,.) =
0.29377022      1.3499486

(2,1,.,.) =
1.3868454       2.3057284

(2,2,.,.) =
2.8272789       -0.01623243

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
[[[[0.58589442 0.94643201 0.24779969 0.55347075]
   [0.50604116 0.69884915 0.81253572 0.58586743]
   [0.94560389 0.11573268 0.12562681 0.63301697]]

  [[0.11736968 0.75641404 0.19342809 0.37670404]
   [0.55561582 0.54354621 0.9506264  0.65929266]
   [0.72911388 0.00499644 0.24280364 0.28822998]]]


 [[[0.53249492 0.43969012 0.20407128 0.49541971]
   [0.00369797 0.75294821 0.15204289 0.41394393]
   [0.19416915 0.93034988 0.0358259  0.38001445]]

  [[0.88946341 0.30646232 0.5347175  0.87568066]
   [0.00439823 0.97792811 0.34842225 0.20433116]
   [0.42777728 0.93583737 0.54341935 0.31203758]]]]

```
Output is:
```python
[[[[0.946432   0.8125357 ]]

  [[0.75641406 0.95062643]]]


 [[[0.7529482  0.4954197 ]]

  [[0.9779281  0.8756807 ]]]]

```

---
## **AveragePooling3D**
Applies average pooling operation for 3D data (spatial or spatio-temporal).

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

Border mode currently supported for this layer is 'valid'.

The input is 5D tensor with shape:(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels).

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
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.AveragePooling3D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(AveragePooling3D[Float](inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
-0.71569425     -0.39595184     -0.47607258
-0.12621938     -0.66759187     0.86833215

(1,1,2,.,.) =
1.219894        -0.07514859     0.6606987
0.073906526     -1.2547257      -0.49249622

(1,2,1,.,.) =
-1.0730773      0.2780401       -0.8603222
-0.31499937     0.94786566      -1.6953986

(1,2,2,.,.) =
0.31038517      1.7660809       -0.9849316
-1.5245554      0.24002236      0.473947

(2,1,1,.,.) =
-0.988634       -0.0028023662   -2.1534977
0.58303267      0.72106487      0.22115333

(2,1,2,.,.) =
1.3964092       -0.59152335     -0.6552192
2.0191588       -0.32599944     0.84014076

(2,2,1,.,.) =
1.4505147       -2.4253457      -0.37597662
-0.7049585      1.3384854       -1.1081233

(2,2,2,.,.) =
-0.8498942      1.169977        0.78120154
0.13814813      -0.7438999      -0.9272572

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
-0.24269137

(1,2,1,.,.) =
0.07872025

(2,1,1,.,.) =
0.3513383

(2,2,1,.,.) =
-0.078371644


[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x1x1x1]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import AveragePooling3D

model = Sequential()
model.add(AveragePooling3D(input_shape = (2, 2, 2, 3)))
input = np.random.random([2, 2, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[[[0.95796698 0.76067104 0.47285625]
    [0.90296063 0.64177821 0.23302549]]

   [[0.37135542 0.38455108 0.66999497]
    [0.06756778 0.16411331 0.39038159]]]


  [[[0.9884323  0.97861344 0.69852249]
    [0.53289779 0.51290587 0.54822396]]

   [[0.77241923 0.06470524 0.00757586]
    [0.65977832 0.31973607 0.7551191 ]]]]



 [[[[0.56819589 0.20398916 0.26409867]
    [0.81165023 0.65269175 0.16519667]]

   [[0.7350688  0.52442381 0.29116889]
    [0.45458689 0.29734681 0.39667421]]]


  [[[0.33577239 0.54035235 0.41285576]
    [0.01023886 0.23677996 0.18901205]]

   [[0.67638612 0.54170351 0.0068781 ]
    [0.95769069 0.88558419 0.4262852 ]]]]]
```
Output is:
```python
[[[[[0.5313706 ]]]


  [[[0.603686  ]]]]



 [[[[0.5309942 ]]]


  [[[0.52306354]]]]]

```

---
## **GlobalMaxPooling1D**
Global max pooling operation for temporal data.

The input is 3D with the shape:(batch_size, steps, features).

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
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.GlobalMaxPooling1D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GlobalMaxPooling1D[Float](inputShape = Shape(2, 3)))
val input = Tensor[Float](2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.2998451       2.1855159       -0.05535197
-0.6448657      0.74119943      -0.8761581

(2,.,.) =
1.3994918       -1.5119147      -0.6625015
1.803635        -2.2516544      -0.016894706

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
1.2998451       2.1855159       -0.05535197
1.803635        -1.5119147      -0.016894706
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import GlobalMaxPooling1D

model = Sequential()
model.add(GlobalMaxPooling1D(input_shape = (2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.05589183 0.73674405 0.49270549]
  [0.03348098 0.82000941 0.81752936]]

 [[0.97310222 0.8878789  0.72330625]
  [0.86144601 0.88568162 0.47241316]]]
```
Output is:
```python
[[0.05589183 0.8200094  0.8175294 ]
 [0.9731022  0.8878789  0.72330624]]
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