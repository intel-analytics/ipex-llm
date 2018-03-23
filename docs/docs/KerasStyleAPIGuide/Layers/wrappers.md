## **TimeDistributed**
TimeDistributed wrapper. Apply a layer to every temporal slice of an input.

The input should be at least 3D, and the dimension of index one will be considered to be the temporal dimension.

**Scala:**
```scala
TimeDistributed(layer, inputShape = null)
```
**Python:**
```python
TimeDistributed(layer, input_shape=None, name=None)
```

Parameters:

* `layer`: A layer instance.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, TimeDistributed, Dense}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(TimeDistributed(Dense(8, activation = "relu"), inputShape = Shape(2, 3)))
val input = Tensor[Float](2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.15650798	-0.60011286	-0.0883946
-0.8020574	-2.0070791	0.58417106

(2,.,.) =
1.1210757	0.061217457	0.37585327
0.11572507	0.045938224	-1.1890792

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.35345355	0.019948795 0.0	        0.22901565	 0.0  0.035260748  0.0	        0.40403664
1.4793522	0.803728	0.0	        0.93547887	 0.0  0.097175285  0.0	        1.2386305

(2,.,.) =
0.06176605	0.0	        0.051847294 0.76588714   0.0  0.67298067   0.10942559   0.0
0.0	        0.0	        0.0	        0.0	         0.0  0.0	       0.4285032    0.3072814

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x8]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import TimeDistributed, Dense

model = Sequential()
model.add(TimeDistributed(Dense(8, activation = "relu"), input_shape = (2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.37107995 0.16777911 0.07691505]
  [0.42678424 0.53602176 0.01580607]]

 [[0.31664302 0.03947526 0.1556008 ]
  [0.2834384  0.68845104 0.23020768]]]
```
Output is:
```python
[[[0.09678233 0.21351711 0.0   0.07420383 0.09885262 0.0 0.13514107 0.0 ]
  [0.06882857 0.18277436 0.0   0.1371126  0.00853634 0.0 0.1224944  0.0 ]]

 [[0.11387025 0.20642482 0.0   0.04896355 0.11478973 0.0 0.12610494 0.0 ]
  [0.08322716 0.08292685 0.0   0.14674747 0.0        0.0 0.05299555 0.0 ]]]
```

---
## **Bidirectional**
Bidirectional wrapper for RNNs.

Bidirectional currently requires RNNs to return the full sequence, i.e. returnSequences = true.

**Scala:**
```scala
Bidirectional(layer, mergeMode = "concat", inputShape = null)
```
**Python:**
```python
Bidirectional(layer, merge_mode="concat", input_shape=None, name=None)
```

Parameters:

* `layer`: An instance of a recurrent layer.
* `mergeMode`: Mode by which outputs of the forward and backward RNNs will be combined. Must be one of: 'sum', 'mul', 'concat', 'ave'. Default is 'concat'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.keras.{Sequential, Bidirectional, SimpleRNN}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Bidirectional(SimpleRNN(4, returnSequences = true), inputShape = Shape(2, 3)))
val input = Tensor[Float](2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.6564635	1.525706	-0.54619956
0.67109746	-0.45657027	-0.5378798

(2,.,.) =
0.19413045	-0.08337678	-0.0016114949
0.6112209	0.7706432	1.3831

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.7013748	0.4841168	  0.10397806 0.3799655   0.6934304	0.27561978	0.44025457	0.44310626
0.4784317	-0.040266205  0.6599038	 -0.29032442 0.55478245	0.061714854	0.5239438	-0.2890968

(2,.,.) =
0.32227796	0.23023699	0.34051302	-0.18683606	0.38275728	0.49924713	0.3152017	-0.14768216
0.1766845	0.39446256	-0.12303881	0.08089487	0.08701726	0.46380803	-0.3540904	-0.0030886582

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x8]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import Bidirectional, LSTM

model = Sequential()
model.add(Bidirectional(LSTM(4, return_sequences = True), merge_mode = "sum", input_shape = (3, 3)))
input = np.random.random([2, 3, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.95180543 0.87111702 0.08901385]
  [0.77432517 0.27843224 0.83308397]
  [0.9140173  0.28253884 0.01381966]]

 [[0.12674146 0.74173106 0.86059416]
  [0.40666387 0.85293504 0.9403338 ]
  [0.42748364 0.14310765 0.98098256]]]
```
Output is:
```python
[[[ 0.11651072  0.07040063  0.53200144 -0.37872505]
  [ 0.03238479  0.15081021  0.55530167 -0.3390156 ]
  [ 0.18388109  0.02891854  0.5591757  -0.28601688]]

 [[-0.17779878 -0.02685877  0.244566   -0.34734237]
  [-0.17816684  0.077871    0.3195565  -0.40989208]
  [-0.13442594  0.08941883  0.3418655  -0.29824993]]]
```
