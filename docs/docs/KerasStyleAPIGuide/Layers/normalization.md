## **BatchNormalization**
Batch normalization layer.

Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.

It is a feature-wise normalization, each feature map in the input will be normalized separately.

The input of this layer should be 4D.

**Scala:**
```scala
BatchNormalization(epsilon = 0.001, momentum = 0.99, betaInit = "zero", gammaInit = "one", dimOrdering = "th", inputShape = null)
```
**Python:**
```python
BatchNormalization(epsilon=0.001, momentum=0.99, beta_init="zero", gamma_init="one", dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `epsilon`: Fuzz parameter. Default is 0.001.
* `momentum`: Momentum in the computation of the exponential average of the mean and standard deviation of the data, for feature-wise normalization. Default is 0.99.
* `betaInit`: Name of initialization function for shift parameter. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'zero'.
* `gammaInit`: Name of initialization function for scale parameter. See [here](initialization/#available-initialization-methods) for available initialization strings. Default is 'one'.
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'. For 'th', axis along which to normalize is 1. For 'tf', axis is 3.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, BatchNormalization}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(BatchNormalization(inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.77107763	  0.2937704	   0.5191167	1.7458088
0.6895759	  1.1034386	   0.076277375	0.73515415
0.8190946	  0.63958114   0.5226141	-0.42864776

(1,2,.,.) =
-0.121818945  0.34588146   0.055290654	-0.07994603
0.6463561	  0.13930246   1.5822772	0.5089318
-0.21778189	  -1.4048384   0.47113693	0.7929269

(2,1,.,.) =
0.6308846	  -0.3855579   1.1685323	1.5646453
0.06638282	  -1.7852567   2.5698936	0.54044205
1.020025	  0.9537036	   -0.95600724	2.0834947

(2,2,.,.) =
-0.5315871	  -1.5204562   -0.19082998	-1.5210537
0.35849532	  0.15615761   -0.55561566	1.1889576
-0.16226959	  -2.1243215   1.1446979	-1.1057223

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.16141568	  -0.3597713   -0.1137085	1.2257558
0.07242136	  0.5243313	   -0.5972589	0.12218969
0.21384695	  0.017830472  -0.10988956	-1.1486028

(1,2,.,.) =
-0.03555677	  0.4775637	   0.15875259	0.010382571
0.80721855	  0.25092307   1.8340303	0.6564485
-0.14083901	  -1.4431748   0.6149832	0.96802336

(2,1,.,.) =
0.008334424	  -1.1015517   0.5954091	1.0279375
-0.6080631	  -2.6299274   2.1256003	-0.090422675
0.4332493	  0.36083078   -1.7244436	1.5944856

(2,2,.,.) =
-0.48511901	  -1.5700207   -0.11126971	-1.5706762
0.49140254	  0.26941508   -0.51148105	1.402514
-0.07993572	  -2.2325294   1.3539561	-1.1150105

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import BatchNormalization

model = Sequential()
model.add(BatchNormalization(input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.01633039 0.08359466 0.31828698 0.31132638]
   [0.82236941 0.34455877 0.40301781 0.09545177]
   [0.32995004 0.21716768 0.40654485 0.0607145 ]]
  [[0.04502162 0.90428985 0.54087212 0.78525733]
   [0.02355475 0.86309013 0.25354746 0.88168388]
   [0.77375427 0.74295181 0.43970331 0.07890251]]]

 [[[0.87290131 0.15790927 0.25248005 0.56290773]
   [0.47154244 0.98287739 0.59877866 0.3287331 ]
   [0.0048165  0.47392756 0.32070177 0.51298559]]
  [[0.89172586 0.68240756 0.86829594 0.79287212]
   [0.13308157 0.04279427 0.59920687 0.26807939]
   [0.42409288 0.54029318 0.65308363 0.90739643]]]]
```
Output is
```python
[[[[-1.3824786   -1.1216924   -0.21178117  -0.2387677 ]
   [ 1.7425659   -0.10992443  0.11672354   -1.075722  ]
   [-0.16656308  -0.6038247   0.13039804   -1.2103996 ]]
  [[-1.6169451   1.149055     -0.02079336  0.7658872 ]
   [-1.6860473   1.0164324    -0.9456966   1.076286  ]
   [ 0.7288585   0.6297049    -0.3464575   -1.5078819 ]]]

 [[[ 1.93848     -0.8335718   -0.4669172   0.73662305]
   [ 0.3823962   2.3648615    0.87569594   -0.17128123]
   [-1.4271184   0.39164335   -0.20241894  0.5430728 ]]
  [[ 1.1086112   0.43481186   1.0331899    0.7903993 ]
   [-1.3334786   -1.6241149   0.16698733   -0.89891803]
   [-0.39670774  -0.02265698  0.3404176    1.1590551 ]]]]
```
