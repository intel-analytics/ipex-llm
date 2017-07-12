## CSubTable ##

**Scala:**
```scala
val model = CSubTable()
```
**Python:**
```python
model = CSubTable()
```

Takes a sequence with two Tensor and returns the component-wise subtraction between them.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

val model = CSubTable()
val input1 = Tensor(5).rand()
val input2 = Tensor(5).rand()
val input = T(input1, input2)
val output = model.forward(input)

scala> print(input)
 {
	2: 0.29122078
	   0.17347474
	   0.14127742
	   0.2249051
	   0.12171601
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]
	1: 0.6202152
	   0.70417005
	   0.21334995
	   0.05191216
	   0.4209623
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]

scala> print(output)
0.3289944
0.5306953
0.072072536
-0.17299294
0.2992463
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
```

**Python example:**
```python
model = CSubTable()
input1 = np.random.randn(5)
input2 = np.random.randn(5)
input = [input1, input2]
output = model.forward(input)
```
output is
```
array([-1.15087152,  0.6169951 ,  2.41840839,  1.34374809,  1.39436531], dtype=float32)
```
