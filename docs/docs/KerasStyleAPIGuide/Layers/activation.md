## **Activation**
Simple activation function to be applied to the output.

**Scala:**
```scala
Activation(activation, inputShape = null)
```
**Python:**
```python
Activation(activation, input_shape=None, name=None)
```

Parameters:

* `activation`: Name of the activation function as string. See [here](#available-activations) for available activation strings.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.Activation
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(Activation[Float]("tanh", inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
2.1659365	0.28006053	-0.20148286
0.9146865	 3.4301455	  1.0930616
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
0.9740552	 0.2729611	  -0.1988
 0.723374	0.99790496	0.7979928
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import Activation
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(Activation("tanh", input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[ 0.26202468  0.15868397  0.27812652]
 [ 0.45931689  0.32100054  0.51839282]]
```
Output is
```python
[[ 0.2561883   0.15736534  0.27117023]
 [ 0.42952728  0.31041133  0.47645861]]
```

Note that the following two pieces of code will be equivalent:
```python
model.add(Dense(32))
model.add(Activation('relu'))
```
```python
model.add(Dense(32, activation="relu"))
```


---
## **Available Activations**
* [elu](#elu)
* [rrelu](#rrelu)
* relu : ReLU applies the element-wise rectified linear unit (ReLU) function to the input.
* selu : Scaled Exponential Linear Unit (SELU). SELU is equal to: scale * elu(x, alpha), where alpha and scale are pre-defined constants. The values of alpha and scale are chosen so that the mean and variance of the inputs are preserved between two consecutive layers as long as the weights are initialized correctly (see lecun_normal initialization) and the number of inputs is "large enough" (see references for more information).
* tanh : Applies the Tanh function element-wise to the input Tensor, thus outputting a Tensor of the same dimension.
* [hardtanh](#hardtanh)
* sigmoid : Applies the Sigmoid function element-wise to the input Tensor, thus outputting a Tensor of the same dimension.
* hard_sigmoid
* softmax : Applies the SoftMax function to an n-dimensional input Tensor, rescaling them so that the
            elements of the n-dimensional output Tensor lie in the range (0, 1) and sum to 1.
            Softmax is defined as:`f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)`
            where `shift = max_i(x_i)`.
* softplus : Apply the SoftPlus function to an n-dimensional input tensor.
* softsign : SoftSign applies SoftSign function to the input tensor
* exponential : Exponential (base e) activation function.
* linear : Linear (i.e. identity) activation function.

---
## ELU ##
Applies exponential linear unit (`ELU`), which parameter a varies the convergence value of the exponential function below zero:

`ELU` is defined as:

```
f(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
```

The output dimension is always equal to input dimension.

For reference see [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/abs/1511.07289).

**Scala:**
```scala
ELU(alpha = 1.0, inputShape = null)
```
**Python:**
```python
m = ELU(alpha=1.0, input_shape=None, name=None)
```

**Parameters:**

* `alpha`: Double, scale for the negative factor. Default is 1.0.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.
 
**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.ELU
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(ELU[Float](1.2, inputShape = Shape(4, 5)))
val input = Tensor[Float](2, 4, 5).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-1.3208098      -0.3994111      1.5678865       -0.5417255      -0.72367394
-0.16772668     -0.28669843     1.0305564       0.15613572      0.29151332
-1.1018531      -0.32264477     -1.4345981      -0.4781121      -2.1548445
0.29493016      1.147811        0.8544963       0.15185815      0.6745268

(2,.,.) =
1.0066849       0.5372675       -0.4647158      -0.64999336     0.97413754
1.0128744       -0.3654132      0.15322192      1.048261        0.9095614
-0.6602698      0.2848114       -0.35451657     -1.3011501      0.7933063
-1.5871915      -0.9177772      0.4741297       0.34224162      -2.7270272

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x5]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.8796972      -0.3951421      1.5678865       -0.5019077      -0.61803937
-0.18529809     -0.29911432     1.0305564       0.15613572      0.29151332
-0.8012943      -0.33092272     -0.9141467      -0.4560568      -1.0608946
0.29493016      1.147811        0.8544963       0.15185815      0.6745268

(2,.,.) =
1.0066849       0.5372675       -0.4460236      -0.5735409      0.97413754
1.0128744       -0.36730814     0.15322192      1.048261        0.9095614
-0.57994574     0.2848114       -0.358185       -0.8733378      0.7933063
-0.9546011      -0.720713       0.4741297       0.34224162      -1.1215038

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4x5]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import ELU
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(ELU(1.2, input_shape=(4, 5)))
input = np.random.random([2, 4, 5])
output = model.forward(input)
```
Input is:
```python
array([[[0.23377795, 0.63399382, 0.20220825, 0.04624555, 0.27801075],
        [0.03957081, 0.35381371, 0.79261921, 0.99816918, 0.16381956],
        [0.49612051, 0.26899042, 0.2938966 , 0.33734888, 0.38244748],
        [0.49566264, 0.32071271, 0.91188529, 0.28086761, 0.8337798 ]],

       [[0.38511148, 0.9840061 , 0.24044046, 0.27135255, 0.89603108],
        [0.60468387, 0.30496216, 0.87750968, 0.56073388, 0.74250063],
        [0.63637121, 0.79358453, 0.26458867, 0.19688831, 0.825432  ],
        [0.14432605, 0.71667083, 0.54347079, 0.82549804, 0.82994232]]])
```
Output is
```python
array([[[0.23377796, 0.6339938 , 0.20220825, 0.04624555, 0.27801076],
        [0.03957081, 0.3538137 , 0.7926192 , 0.9981692 , 0.16381957],
        [0.4961205 , 0.26899043, 0.29389662, 0.33734888, 0.38244748],
        [0.49566263, 0.32071272, 0.91188526, 0.2808676 , 0.8337798 ]],

       [[0.38511148, 0.9840061 , 0.24044046, 0.27135256, 0.8960311 ],
        [0.6046839 , 0.30496216, 0.87750965, 0.5607339 , 0.74250066],
        [0.6363712 , 0.7935845 , 0.26458865, 0.19688831, 0.825432  ],
        [0.14432605, 0.7166708 , 0.5434708 , 0.82549804, 0.82994235]]],
      dtype=float32)
```

---
## **RReLU**
Applies the randomized leaky rectified linear unit element-wise to the input.

f(x) = max(0,x) + a * min(0, x) where a ~ U(l, u).

In the training mode, negative inputs are multiplied by a factor drawn from a uniform random distribution U(l, u).

In the evaluation mode, a RReLU behaves like a LeakyReLU with a constant mean factor a = (l + u) / 2.

If l == u, a RReLU essentially becomes a LeakyReLU.

Regardless of operating in in-place mode a RReLU will internally allocate an input-sized noise tensor to store random factors for negative inputs.

For reference, see [Empirical Evaluation of Rectified Activations in Convolutional Network](http://arxiv.org/abs/1505.00853).

**Scala:**
```scala
RReLU(lower = 1.0/8, upper = 1.0/3, inputShape = null)
```
**Python:**
```python
RReLU(lower=1.0/8, upper=1.0/3, input_shape=None, name=None)
```

**Parameters:**

* `lower`: Lower boundary of the uniform random distribution. Default is 1.0/8.
* `upper`: Upper boundary of the uniform random distribution. Default is 1.0/3.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.RReLU
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(RReLU[Float](inputShape = Shape(1, 4)))
val input = Tensor[Float](1, 1, 4).rand()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.1308445       0.001281989     0.13936701      0.21237929

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.1308445       0.001281989     0.13936701      0.21237929

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x1x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import RReLU
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(RReLU(input_shape = (1, 4)))
input = np.random.random([1, 1, 4])
output = model.forward(input)
```
Input is:
```python
array([[[0.42103899, 0.5255088 , 0.70384155, 0.55685647]]])
```
Ouput is:
```python
array([[[0.421039  , 0.5255088 , 0.70384157, 0.55685645]]], dtype=float32)
```

---
## **HardTanh**
Applies the hard tanh function element-wise to the input.

f(x) = maxValue, if x > maxValue

f(x) = minValue, if x < minValue

f(x) = x, otherwise

**Scala:**
```scala
HardTanh(minValue = -1, maxValue = 1, inputShape = null)
```
**Python:**
```python
HardTanh(min_value=-1, max_value=1, input_shape=None, name=None)
```

**Parameters:**

* `minValue`: The minimum threshold value. Default is -1.
* `maxValue`: The maximum threshold value. Default is 1.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.HardTanh
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(HardTanh[Float](-1, 0.5, inputShape = Shape(3, 4)))
val input = Tensor[Float](2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.8396661       -2.096241       -0.36010137     -1.97987
-0.20326108     1.5972694       -1.4166505      -0.3369559
-0.22637285     -1.1021988      1.0707928       -1.5014135

(2,.,.) =
-0.24511681     -1.1103313      -0.7901563      -1.0394055
-0.033373486    0.22657289      -0.7928737      1.5241393
0.49224186      -0.21418595     -0.32379007     -0.941034

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.5     -1.0    -0.36010137     -1.0
-0.20326108     0.5     -1.0    -0.3369559
-0.22637285     -1.0    0.5     -1.0

(2,.,.) =
-0.24511681     -1.0    -0.7901563      -1.0
-0.033373486    0.22657289      -0.7928737      0.5
0.49224186      -0.21418595     -0.32379007     -0.941034

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import HardTanh
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(HardTanh(-1, 0.5, input_shape=(3, 4)))
input = np.random.random([2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.38707977, 0.94085094, 0.50552125, 0.42818523],
  [0.5544486 , 0.36521357, 0.42551631, 0.93228245],
  [0.29155494, 0.61710319, 0.93137551, 0.05688166]],

 [[0.75222706, 0.36454257, 0.83076327, 0.82004643],
  [0.29213453, 0.71532663, 0.99556398, 0.57001469],
  [0.58088671, 0.32646428, 0.60736   , 0.14861018]]]
```
Output is
```python
[[[0.38707978, 0.5       , 0.5       , 0.42818522],
  [0.5       , 0.36521357, 0.4255163 , 0.5       ],
  [0.29155496, 0.5       , 0.5       , 0.05688166]],

 [[0.5       , 0.36454257, 0.5       , 0.5       ],
  [0.29213452, 0.5       , 0.5       , 0.5       ],
  [0.5       , 0.3264643 , 0.5       , 0.14861017]]]
```