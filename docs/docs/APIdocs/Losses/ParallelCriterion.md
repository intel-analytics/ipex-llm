## ParallelCriterion ##

**Scala:**

```scala
val pc = ParallelCriterion(repeatTarget=false)
```

**Python:**

```python
pc = ParallelCriterion(repeat_target=False)
```

ParallelCriterion is a weighted sum of other criterions each applied to a different input
and target. Set repeatTarget = true to share the target for criterions.
Use add(criterion[, weight]) method to add criterion. Where weight is a scalar(default 1).

**Scala example:**

```scala
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.{Tensor, Storage}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn.{ParallelCriterion, ClassNLLCriterion, MSECriterion}

val pc = ParallelCriterion()

val input = T(Tensor(2, 10), Tensor(2, 10))
var i = 0
input[Tensor](1).apply1(_ => {i += 1; i})
input[Tensor](2).apply1(_ => {i -= 1; i})
val target = T(Tensor(Storage(Array(1.0f, 8.0f))), Tensor(2, 10).fill(1.0f))

val nll = ClassNLLCriterion()
val mse = MSECriterion()
pc.add(nll, 0.5).add(mse)

val output = pc.forward(input, target)
val gradInput = pc.backward(input, target)

println(output)
println(gradInput)

```


The output is,

```
100.75

```

The gradInput is,

```
 {
        2: 1.8000001    1.7     1.6     1.5     1.4     1.3000001       1.2     1.1     1.0     0.90000004
           0.8  0.7     0.6     0.5     0.4     0.3     0.2     0.1     0.0     -0.1
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x10]
        1: -0.25        0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0
           0.0  0.0     0.0     0.0     0.0     0.0     0.0     -0.25   0.0     0.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x10]
 }

```
**Python example:**

```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

pc = ParallelCriterion()

input1 = np.arange(1, 21, 1).astype("float32")
input2 = np.arange(0, 20, 1).astype("float32")[::-1]
input1 = input1.reshape(2, 10)
input2 = input2.reshape(2, 10)

input = [input1, input2]

target1 = np.array([1.0, 8.0]).astype("float32")
target1 = target1.reshape(2)
target2 = np.full([2, 10], 1).astype("float32")
target2 = target2.reshape(2, 10)
target = [target1, target2]

nll = ClassNLLCriterion()
mse = MSECriterion()

pc.add(nll, weight = 0.5).add(mse)

print "input = \n %s " % input
print "target = \n %s" % target

output = pc.forward(input, target)
gradInput = pc.backward(input, target)

print "output = %s " % output
print "gradInput = %s " % gradInput
```

The console will output,

```
input = 
 [array([[  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.],
       [ 11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.]], dtype=float32), array([[ 19.,  18.,  17.,  16.,  15.,  14.,  13.,  12.,  11.,  10.],
       [  9.,   8.,   7.,   6.,   5.,   4.,   3.,   2.,   1.,   0.]], dtype=float32)] 
target = 
 [array([ 1.,  8.], dtype=float32), array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]], dtype=float32)]
output = 100.75 
gradInput = [array([[-0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.25,  0.  ,  0.  ]], dtype=float32), array([[ 1.80000007,  1.70000005,  1.60000002,  1.5       ,  1.39999998,
         1.30000007,  1.20000005,  1.10000002,  1.        ,  0.90000004],
       [ 0.80000001,  0.69999999,  0.60000002,  0.5       ,  0.40000001,
         0.30000001,  0.2       ,  0.1       ,  0.        , -0.1       ]], dtype=float32)]
```
