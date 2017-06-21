## TimeDistributed ##

**Scala:**
```scala
val layer = TimeDistributed[T](layer)
```
**Python:**
```python
layer = TimeDistributed(layer)
```

This layer is intended to apply contained layer to each temporal time slice
of input tensor.

The input data format is [Batch, Time, Other dims]. For the contained layer, it must not change
the Other dims length.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor

val layer = TimeDistributed[Float](Sum[Float](1, squeeze = false, nInputDims = 2))
val input = Tensor[Float](T(T(
  T(
    T(1.0f, 2.0f),
    T(3.0f, 4.0f)
  ),
  T(
    T(2.0f, 3.0f),
    T(4.0f, 5.0f)
  )
)))
layer.forward(input)
layer.backward(input, Tensor[Float](T(T(
  T(
    T(0.1f, 0.2f)
  ),
  T(
    T(0.3f, 0.4f)
  )
))))
```

Its output should be
```
(1,1,.,.) =
4.0     6.0

(1,2,.,.) =
6.0     8.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x1x2]

(1,1,.,.) =
0.1     0.2
0.1     0.2

(1,2,.,.) =
0.3     0.4
0.3     0.4

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x2x2]

```

**Python example:**
```python
Python Code
```