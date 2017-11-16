## Dropout ##

**Scala:**
```scala
val module = Dropout(
  initP = 0.5,
  inplace = false,
  scale = true)
```
**Python:**
```python
module = Dropout(
  init_p=0.5,
  inplace=False,
  scale=True)
```

Dropout masks(set to zero) parts of input using a Bernoulli distribution.
Each input element has a probability `initP` of being dropped. If `scale` is
true(true by default), the outputs are scaled by a factor of `1/(1-initP)` during training.
During evaluating, output is the same as input.

It has been proven an effective approach for regularization and preventing
co-adaptation of feature detectors. For more details, please see
[Improving neural networks by preventing co-adaptation of feature detectors]
(https://arxiv.org/abs/1207.0580)

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val module = Dropout()
val x = Tensor.range(1, 8, 1).resize(2, 4)

println(module.forward(x))
println(module.backward(x, x.clone().mul(0.5f))) // backward drops out the gradients at the same location.
```
Output is
```
com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.0     4.0     6.0     0.0
10.0    12.0    0.0     16.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]

com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.0    2.0    3.0    0.0
5.0    6.0    0.0    8.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Dropout()
x = np.arange(1, 9, 1).reshape(2, 4)

print(module.forward(x))
print(module.backward(x, x.copy() * 0.5)) # backward drops out the gradients at the same location.
```
Output is
```
[array([[ 0.,  4.,  6.,  0.],
       [ 0.,  0.,  0.,  0.]], dtype=float32)]
       
[array([[ 0.,  2.,  3.,  0.],
       [ 0.,  0.,  0.,  0.]], dtype=float32)]
```


## GaussianDropout

**Scala:**
```scala
val module = GaussianDropout(rate)
```
**Python:**
```python
module = GaussianDropout(rate)
```

Apply multiplicative 1-centered Gaussian noise.
As it is a regularization layer, it is only active at training time.

* `rate` is drop probability (as with `Dropout`).

Reference: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

**Scala example:**
```scala
scala> layer.training()
res0: layer.type = GaussianDropout[fa8b4a5](0.5)

scala> val input = Tensor(T(T(1.0,1.0,1.0),T(1.0,1.0,1.0)))
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0     1.0     1.0
1.0     1.0     1.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

scala> val output = layer.forward(input)
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.6049535      1.6871624       0.3253751
1.2154467       1.4057124       2.019263
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

scala> val gradout = Tensor(T(T(1.0,1.0,1.0),T(1.0,1.0,1.0)))
gradout: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0     1.0     1.0
1.0     1.0     1.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

scala> val gradin = layer.backward(input,gradout)
gradin: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.6049535      1.6871624       0.3253751
1.2154467       1.4057124       2.019263
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

scala> layer.evaluate()
res1: layer.type = GaussianDropout[fa8b4a5](0.5)

scala> val output = layer.forward(input)
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0     1.0     1.0
1.0     1.0     1.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]
```

**Python example:**
```python
```
Output is
```
```

## GaussianNoise

**Scala:**
```scala
val module = GaussianNoise(stddev)
```
**Python:**
```python
module = GaussianNoise(stddev)
```

Apply additive zero-centered Gaussian noise. This is useful to mitigate overfitting (you could see it as a form of random data augmentation).
Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs.

As it is a regularization layer, it is only active at training time.

* `stddev` is the standard deviation of the noise distribution.

**Scala example:**
```scala
```
Output is
```
```

**Python example:**
```python
```
Output is
```
```
