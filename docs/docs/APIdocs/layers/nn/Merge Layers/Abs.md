## Abs ##

**Scala:**
```scala
abs = Abs()
```
**Python:**
```python
abs = Abs()
```

An element-wise abs operation.


**Scala example:**
```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val abs = new Abs
val input = Tensor(2)
input(1) = 21f
input(2) = -29f
print(abs.forward(input))
```
`output is:　21.0　29.0`

**Python example:**
```python
abs = Abs()
input = np.array([21, -29, 30])
print(abs.forward(input))
```
`output is: [array([ 21.,  29.,  30.], dtype=float32)]`

