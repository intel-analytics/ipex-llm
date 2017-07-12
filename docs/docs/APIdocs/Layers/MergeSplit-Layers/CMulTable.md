## CMulTable ##

**Scala:**
```scala
val model = CMulTable()
```
**Python:**
```python
model = CMulTable()
```

Takes a sequence of Tensors and outputs the multiplication of all of them.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

val model = CMulTable()
val input1 = Tensor(5).rand()
val input2 = Tensor(5).rand()
val input = T(input1, input2)
val output = model.forward(input)

scala> print(input)
 {
	2: 0.13224044
	   0.5460452
	   0.33032498
	   0.6317603
	   0.6665052
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]
	1: 0.28694472
	   0.45169437
	   0.36891535
	   0.9126049
	   0.41318864
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]
 }

scala> print(output)
0.037945695
0.24664554
0.12186196
0.57654756
0.27539238
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
```

**Python example:**
```python
model = CMulTable()
input1 = np.random.randn(5)
input2 = np.random.randn(5)
input = [input1, input2]
output = model.forward(input)

>>> print(input)
[array([ 0.28183274, -0.6477487 , -0.21279841,  0.22725124,  0.54748552]), array([-0.78673028, -1.08337196, -0.62710066,  0.37332587, -1.40708162])]

>>> print(output)
[-0.22172636  0.70175284  0.13344601  0.08483877 -0.77035683]
```
