## CSubTable ##

**Scala:**
```scala
val model = CSubTable[T]()
```
**Python:**
```python
model = CSubTable()
```

Takes a table with two Tensor and returns the component-wise subtraction between them.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

val model = CSubTable[Float]()

val input1 = Tensor[Float](5).rand()
val input2 = Tensor[Float](5).rand()
val input = T(input1, input2)
val output = model.forward(input)
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
0.18143779
-0.24954873
-0.42380047
0.083815336
-0.10043772
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
