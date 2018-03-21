---
## **BatchNormalization**
Batch normalization layer.

Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.

It is a feature-wise normalization, each feature map in the input will be normalized separately.

The input of this layer should be 4D.

**Scala:**
```scala
BatchNormalization(epsilon = 0.001, momentum = 0.99, betaInit = "zero", gammaInit = "one", dimOrdering = DataFormat.NCHW, inputShape = null)
```
**Python:**
```python
BatchNormalization(epsilon=0.001, momentum=0.99, beta_init="zero", gamma_init="one", dim_ordering="th", input_shape=None)
```

**Parameters:**

* `epsilon`: Small Double > 0. Fuzz parameter. Default is 0.001.
* `momentum`: Double. Momentum in the computation of the exponential average of the mean and standard deviation of the data, for feature-wise normalization. Default is 0.99.
* `betaInit`: Name of initialization function for shift parameter. Default is 'zero'.
* `gammaInit`: Name of initialization function for scale parameter. Default is 'one'.
* `dimOrdering`: Format of input data. Either DataFormat.NCHW (dimOrdering='th') or DataFormat.NHWC (dimOrdering='tf'). Default is NCHW. For NCHW, axis along which to normalize is 1. For NHWC, axis is 3.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, BatchNormalization}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor._

val model = Sequential[Float]()
model.add(BatchNormalization(betaInit = "glorot_uniform", gammaInit = "normal", inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.42762622	-1.0492688	-0.076253355	1.1452506
0.4923164	-0.9118383	-1.5972881	1.6674142
0.27690548	1.2325531	1.3604681	0.8444599

(1,2,.,.) =
0.9948373	1.2529879	1.9710598	-0.3868922
0.11610765	-0.2942152	1.50801	0.75046474
-2.4630868	-0.99648345	-0.25959244	-0.4335127

(2,1,.,.) =
0.98660356	-0.21924433	-0.22566248	-2.9577894
0.26295888	-0.5073583	0.2803466	0.771863
-0.09193111	1.3429742	-1.2742913	-0.86854964

(2,2,.,.) =
1.4430099	-1.2574246	0.87110335	0.1020681
1.4758815	0.15900438	0.4701082	-0.7687285
0.38702253	-1.5793637	0.9927724	-1.6198405

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
1.0996926	1.1072749	1.1022795	1.0960083
1.0993605	1.1065694	1.1100885	1.0933276
1.1004664	1.0955602	1.0949035	1.0975527

(1,2,.,.) =
0.9847812	0.9869138	0.9928458	0.9733667
0.977522	0.9741323	0.9890205	0.9827624
0.95621526	0.96833086	0.97441834	0.9729816

(2,1,.,.) =
1.0968229	1.1030136	1.1030465	1.1170732
1.100538	1.1044928	1.1004487	1.0979253
1.10236	1.0949932	1.1084301	1.1063471

(2,2,.,.) =
0.98848355	0.9661752	0.98375905	0.977406
0.9887551	0.97787637	0.9804464	0.97021234
0.97976005	0.9635157	0.98476416	0.9631813

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import BatchNormalization

model = Sequential()
model.add(BatchNormalization(input_shape=(3, 3, 4)))
input = np.random.random([2, 3, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.23365801 0.994575   0.4660027  0.49074937]
   [0.50963456 0.82502413 0.83978148 0.61729025]
   [0.64252898 0.27880527 0.91943315 0.95842744]]

  [[0.05032944 0.46030287 0.60433308 0.54666088]
   [0.52151799 0.01745638 0.97687435 0.29189112]
   [0.50405231 0.8823347  0.80198682 0.4711616 ]]

  [[0.3513452  0.22994895 0.34193399 0.52748137]
   [0.10595707 0.0728661  0.7820933  0.04032613]
   [0.30940133 0.53535336 0.36517076 0.77541009]]]


 [[[0.80688988 0.63770275 0.08063848 0.5880161 ]
   [0.41205317 0.40531198 0.80507185 0.24764818]
   [0.00817546 0.39100586 0.06716752 0.96262487]]

  [[0.96366958 0.89893076 0.46232989 0.32278294]
   [0.09654676 0.6742205  0.04698092 0.507851  ]
   [0.19971914 0.8353601  0.67230662 0.56905002]]

  [[0.74075079 0.77001342 0.73738289 0.29069895]
   [0.04840953 0.99938287 0.2326268  0.37940441]
   [0.20654755 0.91972152 0.19581629 0.33459691]]]]
```
Output is
```python
[[[[-1.0683918   1.5054718  -0.282467   -0.19875942]
   [-0.13487871  0.9319521   0.98187     0.22927545]
   [ 0.31464738 -0.9156775   1.2512982   1.3831997 ]]

  [[-1.6045189  -0.19123396  0.30527574  0.10646465]
   [ 0.01979051 -1.7178409   1.5895222  -0.77179295]
   [-0.04041813  1.2636195   0.98663944 -0.15380104]]

  [[-0.2729185  -0.7003371  -0.306054    0.34723145]
   [-1.1368946  -1.2534031   1.2436831  -1.3679717 ]
   [-0.4205968   0.37494758 -0.22424076  1.2201526 ]]]


 [[[ 0.8706116   0.2983224  -1.5859928   0.13025321]
   [-0.4649557  -0.48775834  0.8644619  -1.0210689 ]
   [-1.8311049  -0.53615    -1.6315594   1.3973978 ]]

  [[ 1.544002    1.3208305  -0.18424624 -0.6653009 ]
   [-1.4451958   0.54619575 -1.6160622  -0.02732314]
   [-1.0895338   1.1016859   0.53959805  0.18364576]]

  [[ 1.0981221   1.2011516   1.0862643  -0.4864452 ]
   [-1.3395112   2.008728   -0.69090885 -0.17412604]
   [-0.78273004  1.7282523  -0.8205133  -0.33188686]]]]
```

---
