## Reverse ##


**Scala:**
```scala
val m = Reverse(dim = 1, isInplace = false)
```
**Python:**
```python
m = Reverse(dimension=1, is_inplace=False)
```

 Reverse the input w.r.t given dimension.
 The input can be a Tensor or Table. __Dimension is one-based index__ 
 

**Scala example:**
```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

def randomn(): Float = RandomGenerator.RNG.uniform(0, 1)
val input = Tensor(2, 3)
input.apply1(x => randomn().toFloat)
println("input:")
println(input)
val layer = new Reverse(1)
println("output:")
println(layer.forward(input))
```

```
input:
0.17271264898590744	0.019822501810267568	0.18107921979390085	
0.4003877849318087	0.5567442716564983	0.14120339532382786	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
output:
0.4003877849318087	0.5567442716564983	0.14120339532382786	
0.17271264898590744	0.019822501810267568	0.18107921979390085	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]

```



**Python example:**
```python
input = np.random.random((2,3))
layer = Reverse(1)
print("input:")
print(input)
print("output:")
print(layer.forward(input))
```
```
creating: createReverse
input:
[[ 0.89089717  0.07629756  0.30863782]
 [ 0.16066851  0.06421963  0.96719367]]
output:
[[ 0.16066851  0.06421963  0.96719366]
 [ 0.89089715  0.07629756  0.30863783]]

 
```

