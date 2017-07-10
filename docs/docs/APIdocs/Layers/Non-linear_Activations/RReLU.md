## RReLU ##

**Scala:**
```scala
val layer = RReLU[T](lower, upper, inPlace)
```
**Python:**
```python
layer = RReLU(lower, upper, inPlace)
```

Applies the randomized leaky rectified linear unit (RReLU) element-wise to the input Tensor,
thus outputting a Tensor of the same dimension. Informally the RReLU is also known as 'insanity' layer.

RReLU is defined as: f(x) = max(0,x) + a * min(0, x) where a ~ U(l, u).

In training mode negative inputs are multiplied by a factor drawn from a uniform random
distribution U(l, u). In evaluation mode a RReLU behaves like a LeakyReLU with a constant mean
factor a = (l + u) / 2.

By default, l = 1/8 and u = 1/3. If l == u a RReLU effectively becomes a LeakyReLU.

Regardless of operating in in-place mode a RReLU will internally allocate an input-sized noise tensor to store random factors for negative inputs.

The backward() operation assumes that forward() has been called before.

For reference see [Empirical Evaluation of Rectified Activations in Convolutional Network](http://arxiv.org/abs/1505.00853).

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor

val layer = RReLU[Float]()
layer.forward(Tensor[Float](T(1.0f, 2.0f, -1.0f, -2.0f)))
layer.backward(Tensor[Float](T(1.0f, 2.0f, -1.0f, -2.0f)),
Tensor[Float](T(0.1f, 0.2f, -0.1f, -0.2f)))
```

There's random factor. An output is like
```
1.0
2.0
-0.24342789
-0.43175703
[com.intel.analytics.bigdl.tensor.DenseTensor of size 4]

0.1
0.2
-0.024342788
-0.043175705
[com.intel.analytics.bigdl.tensor.DenseTensor of size 4]
```

**Python example:**
```python
from bigdl.nn.layer import RReLU
import numpy as np

layer = RReLU()
layer.forward(np.array([1.0, 2.0, -1.0, -2.0]))
layer.backward(np.array([1.0, 2.0, -1.0, -2.0]),
  np.array([0.1, 0.2, -0.1, -0.2]))
```

There's random factor. An output is like
```
array([ 1.,  2., -0.15329693, -0.40423378], dtype=float32)

array([ 0.1, 0.2, -0.01532969, -0.04042338], dtype=float32)
```