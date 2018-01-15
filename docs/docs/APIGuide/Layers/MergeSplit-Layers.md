## Pack ##

**Scala:**
```scala
val module = Pack(dim)
```
**Python:**
```python
module = Pack(dim)
```

Pack is used to stack a list of n-dimensional tensors into one (n+1)-dimensional tensor.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val module = Pack(2)
val input1 = Tensor(2, 2).randn()
val input2 = Tensor(2, 2).randn()
val input = T()
input(1) = input1
input(2) = input2

val output = module.forward(input)

> input
 {
	2: -0.8737048	-0.7337217
	   0.7268678	-0.53470045
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
	1: -1.3062215	-0.58756566
	   0.8921608	-1.8087773
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
 }

 
> output
(1,.,.) =
-1.3062215	-0.58756566
-0.8737048	-0.7337217

(2,.,.) =
0.8921608	-1.8087773
0.7268678	-0.53470045

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Pack(2)
input1 = np.random.randn(2, 2)
input2 = np.random.randn(2, 2)
input = [input1, input2]
output = module.forward(input)

> input
[array([[ 0.92741416, -3.29826586],
       [-0.03147819, -0.10049306]]), array([[-0.27146461, -0.25729802],
       [ 0.1316149 ,  1.27620145]])]
       
> output
array([[[ 0.92741418, -3.29826593],
        [-0.27146462, -0.25729802]],

       [[-0.03147819, -0.10049306],
        [ 0.13161489,  1.27620149]]], dtype=float32)
```
---
## MM ##

**Scala:**
```scala
val m = MM(transA=false,transB=false)
```
**Python:**
```python
m = MM(trans_a=False,trans_b=False)
```


MM is a module that performs matrix multiplication on two mini-batch inputs, producing one mini-batch.

**Scala example:**
```scala
scala>
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T

val input = T(1 -> Tensor(3, 3).randn(), 2 -> Tensor(3, 3).randn())
val m1 = MM()
val output1 = m1.forward(input)
val m2 = MM(true,true)
val output2 = m2.forward(input)

scala> print(input)
 {
        2: -0.62020904  -0.18690863     0.34132162
           -0.5359324   -0.09937895     0.86147165
           -2.6607985   -1.426654       2.3428898
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]
        1: -1.3087689   0.048720464     0.69583243
           -0.52055264  -1.5275089      -1.1569321
           0.28093573   -0.29353273     -0.9505267
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3]
 }

scala> print(output1)
-1.0658705      -0.7529337      1.225519
4.2198563       1.8996398       -4.204146
2.512235        1.3327343       -2.38396
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

scala> print(output2)
1.0048954       0.99516183      4.8832207
0.15509865      -0.12717877     1.3618765
-0.5397563      -1.0767963      -2.4279075
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

```
---
**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input1=np.random.rand(3,3)
input2=np.random.rand(3,3)
input = [input1,input2]
print "input is :",input
out = MM().forward(input)
print "output is :",out
```
produces output:
```python
input is : [array([[ 0.13696046,  0.92653165,  0.73585328],
       [ 0.28167852,  0.06431783,  0.15710073],
       [ 0.21896166,  0.00780161,  0.25780671]]), array([[ 0.11232797,  0.17023931,  0.92430042],
       [ 0.86629537,  0.07630215,  0.08584417],
       [ 0.47087278,  0.22992833,  0.59257503]])]
creating: createMM
output is : [array([[ 1.16452789,  0.26320592,  0.64217824],
       [ 0.16133308,  0.08898225,  0.35897085],
       [ 0.15274818,  0.09714822,  0.3558259 ]], dtype=float32)]
```
## CMaxTable ##

**Scala:**
```scala
val m = CMaxTable()
```
**Python:**
```python
m = CMaxTable()
```

CMaxTable is a module that takes a table of Tensors and outputs the max of all of them.


**Scala example:**
```scala

scala> 
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T

val input1 = Tensor(3).randn()
val input2 =  Tensor(3).randn()
val input = T(input1, input2)
val m = CMaxTable()
val output = m.forward(input)
val gradOut = Tensor(3).randn()
val gradIn = m.backward(input,gradOut)

scala> print(input)
 {
        2: -0.38613814
           0.74074316
           -1.753783
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
        1: -1.6037064
           -2.3297918
           -0.7160026
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
 }

scala> print(output)
-0.38613814
0.74074316
-0.7160026
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]

scala> print(gradOut)
-1.4526331
0.7070323
0.29294914
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]

scala> print(gradIn)
 {
        2: -1.4526331
           0.7070323
           0.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
        1: 0.0
           0.0
           0.29294914
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
 }

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input1 = np.random.rand(3)
input2 = np.random.rand(3)
print "input is :",input1,input2

m = CMaxTable()
out = m.forward([input1,input2])
print "output of m is :",out

grad_out = np.random.rand(3)
grad_in = m.backward([input1, input2],grad_out)
print "grad input of m is :",grad_in
```
Gives the output,

```python
input is : [ 0.48649797  0.22131348  0.45667796] [ 0.73207053  0.74290136  0.03169769]
creating: createCMaxTable
output of m is : [array([ 0.73207051,  0.74290138,  0.45667794], dtype=float32)]
grad input of m is : [array([ 0.        ,  0.        ,  0.86938971], dtype=float32), array([ 0.04140199,  0.4787094 ,  0.        ], dtype=float32)]
```
---
## SplitTable ##

**Scala:**
```scala
val layer = SplitTable(dim)
```
**Python:**
```python
layer = SplitTable(dim)
```

SplitTable takes a Tensor as input and outputs several tables,
splitting the Tensor along the specified dimension `dimension`. Please note
the dimension starts from 1.

The input to this layer is expected to be a tensor, or a batch of tensors;
when using mini-batch, a batch of sample tensors will be passed to the layer and
the user needs to specify the number of dimensions of each sample tensor in a
batch using `nInputDims`.

```
    +----------+         +-----------+
    | input[1] +---------> {member1, |
  +----------+-+         |           |
  | input[2] +----------->  member2, |
+----------+-+           |           |
| input[3] +------------->  member3} |
+----------+             +-----------+
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val layer = SplitTable(2)
layer.forward(Tensor(T(
  T(1.0f, 2.0f, 3.0f),
  T(4.0f, 5.0f, 6.0f),
  T(7.0f, 8.0f, 9.0f)
)))
layer.backward(Tensor(T(
  T(1.0f, 2.0f, 3.0f),
  T(4.0f, 5.0f, 6.0f),
  T(7.0f, 8.0f, 9.0f)
)), T(
  Tensor(T(0.1f, 0.2f, 0.3f)),
  Tensor(T(0.4f, 0.5f, 0.6f)),
  Tensor(T(0.7f, 0.8f, 0.9f))
))
```

Gives the output,
```
 {
        2: 2.0
           5.0
           8.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
        1: 1.0
           4.0
           7.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
        3: 3.0
           6.0
           9.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
 }

0.1     0.4     0.7
0.2     0.5     0.8
0.3     0.6     0.9
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
```

**Python example:**
```python
from bigdl.nn.layer import SplitTable
import numpy as np

layer = SplitTable(2)
layer.forward(np.array([
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0],
  [7.0, 8.0, 9.0]
]))

layer.backward(np.array([
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0],
  [7.0, 8.0, 9.0]
]), [
  np.array([0.1, 0.2, 0.3]),
  np.array([0.4, 0.5, 0.6]),
  np.array([0.7, 0.8, 0.9])
])
```
Gives the output,
```
[
  array([ 1.,  4.,  7.], dtype=float32),
  array([ 2.,  5.,  8.], dtype=float32),
  array([ 3.,  6.,  9.], dtype=float32)
]

array([[ 0.1       ,  0.40000001,  0.69999999],
       [ 0.2       ,  0.5       ,  0.80000001],
       [ 0.30000001,  0.60000002,  0.89999998]], dtype=float32)
```
---
## DotProduct ##

**Scala:**

```scala
val m = DotProduct()
```
**Python:**
```python
m = DotProduct()
```

Outputs the dot product (similarity) between inputs


**Scala example:**
```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val mlp = DotProduct()
val x = Tensor(3).fill(1f)
val y = Tensor(3).fill(2f)
println("input:")
println(x)
println(y)
println("output:")
println(mlp.forward(T(x, y)))
```
```
input:
1.0
1.0
1.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
2.0
2.0
2.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
output:
6.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1]

```

**Python example:**

```python
import numpy as np
from bigdl.nn.layer import *

mlp = DotProduct()
x = np.array([1, 1, 1])
y = np.array([2, 2, 2])
print("input:")
print(x)
print(y)
print("output:")
print(mlp.forward([x, y]))

```
```
creating: createDotProduct
input:
[1 1 1]
[2 2 2]
output:
[ 6.]
```

---
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
Gives the output,
```
array([-1.15087152,  0.6169951 ,  2.41840839,  1.34374809,  1.39436531], dtype=float32)
```
---
## CDivTable ##

**Scala:**
```scala
val module = CDivTable()
```
**Python:**
```python
module = CDivTable()
```

Takes a table with two Tensor and returns the component-wise division between them.

**Scala example:**
```scala
val module = CDivTable()
val input = T(1 -> Tensor(2,3).rand(), 2 -> Tensor(2,3).rand())
input: com.intel.analytics.bigdl.utils.Table =
 {
        2: 0.802295     0.7113872       0.29395157
           0.6562403    0.06519115      0.20099664
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]
        1: 0.7435388    0.59126955      0.10225375
           0.46819785   0.10572237      0.9861797
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]
 }

module.forward(input)
res6: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.9267648       0.8311501       0.34785917
0.7134549       1.6217289       4.906449
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
module = CDivTable()
input = [np.array([[1, 2, 3],[4, 5, 6]]), np.array([[1, 4, 9],[6, 10, 3]])]
module.forward(input)
[array([
[ 1.,                   0.5     ,    0.33333334],
[ 0.66666669, 0.5       ,  2.        ]], dtype=float32)]
```
---
## JoinTable
**Scala:**
```scala
val layer = JoinTable(dimension, nInputDims)
```
**Python:**
```python
layer = JoinTable(dimension, n_input_dims)
```

It is a table module which takes a table of Tensors as input and
outputs a Tensor by joining them together along the dimension `dimension`.


The input to this layer is expected to be a tensor, or a batch of tensors;
when using mini-batch, a batch of sample tensors will be passed to the layer and
the user need to specify the number of dimensions of each sample tensor in the
batch using `nInputDims`.

Parameters:
* `dimension`  to be join in this dimension
* `nInputDims` specify the number of dimensions that this module will receiveIf it is more than the dimension of input tensors, the first dimension would be considered as batch size

```
+----------+             +-----------+
| {input1, +-------------> output[1] |
|          |           +-----------+-+
|  input2, +-----------> output[2] |
|          |         +-----------+-+
|  input3} +---------> output[3] |
+----------+         +-----------+
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.JoinTable
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val layer = JoinTable(2, 2)
val input1 = Tensor(T(
  T(
    T(1f, 2f, 3f),
    T(2f, 3f, 4f),
    T(3f, 4f, 5f))
))

val input2 = Tensor(T(
  T(
    T(3f, 4f, 5f),
    T(2f, 3f, 4f),
    T(1f, 2f, 3f))
))

val input = T(input1, input2)

val gradOutput = Tensor(T(
  T(
    T(1f, 2f, 3f, 3f, 4f, 5f),
    T(2f, 3f, 4f, 2f, 3f, 4f),
    T(3f, 4f, 5f, 1f, 2f, 3f)
)))

val output = layer.forward(input)
val grad = layer.backward(input, gradOutput)

println(output)
(1,.,.) =
1.0	2.0	3.0	3.0	4.0	5.0
2.0	3.0	4.0	2.0	3.0	4.0
3.0	4.0	5.0	1.0	2.0	3.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x6]

println(grad)
 {
	2: (1,.,.) =
	   3.0	4.0	5.0
	   2.0	3.0	4.0
	   1.0	2.0	3.0

	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x3]
	1: (1,.,.) =
	   1.0	2.0	3.0
	   2.0	3.0	4.0
	   3.0	4.0	5.0

	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x3]
 }
```

**Python example:**
```python
layer = JoinTable(2, 2)
input1 = np.array([
 [
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0]
  ]
])

input2 = np.array([
  [
    [3.0, 4.0, 5.0],
    [2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0]
  ]
])

input = [input1, input2]

gradOutput = np.array([
  [
    [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
    [2.0, 3.0, 4.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 1.0, 2.0, 3.0]
  ]
])

output = layer.forward(input)
grad = layer.backward(input, gradOutput)

print output
[[[ 1.  2.  3.  3.  4.  5.]
  [ 2.  3.  4.  2.  3.  4.]
  [ 3.  4.  5.  1.  2.  3.]]]

print grad
[array([[[ 1.,  2.,  3.],
        [ 2.,  3.,  4.],
        [ 3.,  4.,  5.]]], dtype=float32), array([[[ 3.,  4.,  5.],
        [ 2.,  3.,  4.],
        [ 1.,  2.,  3.]]], dtype=float32)]
```
---
## SelectTable ##

**Scala:**
```scala
val m = SelectTable(index: Int)
```
**Python:**
```python
m = SelectTable(dimension)
```

Select one element from a table by a given index.
In Scala API, table is kind of like HashMap with one-base index as the key.
In python, table is a just a list.


**Scala example:**

```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val input = T(Tensor(2,3).randn(), Tensor(2,3).randn())

println("input: ")
println(input)
println("output:")
println(SelectTable(1).forward(input)) // Select and output the first element of the input which shape is (2, 3)
println(SelectTable(2).forward(input)) // Select and output the second element of the input which shape is (2, 3)

```
```
input: 
 {
	2: 2.005436370849835	0.09670211785545313	1.186779895312918	
	   2.238415300857082	0.241626512721254	0.15765709974113828	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
	1: 0.5668905654052705	-1.3205159007397167	-0.5431464848526197	
	   -0.11582559521074104	0.7671830693813515	-0.39992781407893574	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
 }
output:
0.5668905654052705	-1.3205159007397167	-0.5431464848526197	
-0.11582559521074104	0.7671830693813515	-0.39992781407893574	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
2.005436370849835	0.09670211785545313	1.186779895312918	
2.238415300857082	0.241626512721254	0.15765709974113828	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]

```

**Python example:**
```python
import numpy as np
from bigdl.nn.layer import *

input = [np.random.random((2,3)), np.random.random((2, 1))]
print("input:")
print(input)
print("output:")
print(SelectTable(1).forward(input)) # Select and output the first element of the input which shape is (2, 3)
```
```
input:
[array([[ 0.07185111,  0.26140439,  0.9437582 ],
       [ 0.50278191,  0.83923974,  0.06396735]]), array([[ 0.84955122],
       [ 0.16053703]])]
output:
creating: createSelectTable
[[ 0.07185111  0.2614044   0.94375819]
 [ 0.50278193  0.83923972  0.06396735]]
 
```

---
## NarrowTable ##

**Scala:**
```scala
val narrowTable = NarrowTable(offset, length = 1)
```
**Python:**
```python
narrowTable = NarrowTable(offset, length = 1)
```

NarrowTable takes a table as input and returns a subtable starting from index `offset` having `length` elements

Negative `length` means the last element is located at Abs|length| to the last element of input

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T
val narrowTable = NarrowTable(1, 1)

val input = T()
input(1.0) = Tensor(2, 2).rand()
input(2.0) = Tensor(2, 2).rand()
input(3.0) = Tensor(2, 2).rand()
> print(input)
 {
	2.0: 0.27686104	0.9040761	
	     0.75969505	0.8008061	
	     [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
	1.0: 0.94122535	0.46173728	
	     0.43302807	0.1670979	
	     [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
	3.0: 0.43944374	0.49336782	
	     0.7274511	0.67777634	
	     [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
 }
>  print(narrowTable.forward(input))
 {
	1: 0.94122535	0.46173728	
	   0.43302807	0.1670979	
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
 }

```

**Python example:**
```python
from bigdl.nn.layer import *
narrowTable = NarrowTable(1, 1)
> narrowTable.forward([np.array([1, 2, 3]), np.array([4, 5, 6])])
[array([ 1.,  2.,  3.], dtype=float32)]
       
```

---
## CAddTable ##

**Scala:**
```scala
val module = CAddTable(inplace = false)
```
**Python:**
```python
module = CAddTable(inplace=False)
```

CAddTable merges the input tensors in the input table by element-wise adding. The input table is actually an array of tensor with same size.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val mlp = Sequential()
mlp.add(ConcatTable().add(Identity()).add(Identity()))
mlp.add(CAddTable())

println(mlp.forward(Tensor.range(1, 3, 1)))
```
Gives the output,
```
com.intel.analytics.bigdl.nn.abstractnn.Activity =
2.0
4.0
6.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

mlp = Sequential()
mlp.add(ConcatTable().add(Identity()).add(Identity()))
mlp.add(CAddTable())

print(mlp.forward(np.arange(1, 4, 1)))
```
Gives the output,
```
[array([ 2.,  4.,  6.], dtype=float32)]
```

---
## CAveTable ##

**Scala:**
```scala
val model = CAveTable(inplace=false)
```
**Python:**
```python
model = CAveTable(inplace=False)
```

CAveTable merges the input tensors in the input table by element-wise taking the average. The input table is actually an array of tensor with same size.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T

val model = CAveTable()
val input1 = Tensor(5).rand()
val input2 = Tensor(5).rand()
val input = T(input1, input2)
val output = model.forward(input)
```
Gives the output,
```
input1: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.6061657
0.55972266
0.972365
0.5624792
0.7495829
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]

input2: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.3897284
0.82165825
0.46275142
0.95935726
0.64157426
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 5]

output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.49794704
0.69069046
0.7175582
0.76091826
0.6955786
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

model = CAveTable()
input1 = np.random.rand(5)
input2 = np.random.rand(5)
input = [input1, input2]
output = model.forward(input)

print(input1)
print(input2)
print(output)
```
Gives the output,
```
[array([ 0.26202468  0.15868397  0.27812652  0.45931689  0.32100054], dtype=float32)]
[array([ 0.51839282  0.26194293  0.97608528  0.73281455  0.11527423], dtype=float32)]
[array([ 0.39020872  0.21031344  0.62710589  0.5960657   0.21813738], dtype=float32)]
```

---
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
---
## MV ##

**Scala:**
```scala
val module = MV(trans = false)
```
**Python:**
```python
module = MV(trans=False)
```

It is a module to perform matrix vector multiplication on two mini-batch inputs, producing a mini-batch.

`trans` means whether make matrix transpose before multiplication.


**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T

val module = MV()

println(module.forward(T(Tensor.range(1, 12, 1).resize(2, 2, 3), Tensor.range(1, 6, 1).resize(2, 3))))
```
Gives the output,
```
com.intel.analytics.bigdl.tensor.Tensor[Float] =
14.0	32.0
122.0	167.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

**Python example:**
```python
module = MV()

print(module.forward([np.arange(1, 13, 1).reshape(2, 2, 3), np.arange(1, 7, 1).reshape(2, 3)]))
```
Gives the output,
```
[array([ 0.31657887, -1.11062765, -1.16235781, -0.67723978,  0.74650359], dtype=float32)]
```
---
## FlattenTable ##

**Scala:**
```scala
val module = FlattenTable()
```
**Python:**
```python
module = FlattenTable()
```

FlattenTable takes an arbitrarily deep table of Tensors (potentially nested) as input and a table of Tensors without any nested table will be produced

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val module = FlattenTable()
val t1 = Tensor(3).randn()
val t2 = Tensor(3).randn()
val t3 = Tensor(3).randn()
val input = T(t1, T(t2, T(t3)))

val output = module.forward(input)

> input
 {
	2:  {
	   	2:  {
	   	   	1: 0.5521984
	   	   	   -0.4160644
	   	   	   -0.698762
	   	   	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
	   	    }
	   	1: -1.7380241
	   	   0.60336906
	   	   -0.8751049
	   	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
	    }
	1: 1.0529885
	   -0.792229
	   0.8395628
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
 }


> output
{
	2: -1.7380241
	   0.60336906
	   -0.8751049
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
	1: 1.0529885
	   -0.792229
	   0.8395628
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
	3: 0.5521984
	   -0.4160644
	   -0.698762
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]
 }

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

module = Sequential()
# this will create a nested table
nested = ConcatTable().add(Identity()).add(Identity())
module.add(nested).add(FlattenTable())
t1 = np.random.randn(3)
t2 = np.random.randn(3)
input = [t1, t2]
output = module.forward(input)

> input
[array([-2.21080689, -0.48928043, -0.26122161]), array([-0.8499716 ,  1.63694575, -0.31109292])]

> output
[array([-2.21080685, -0.48928043, -0.26122162], dtype=float32),
 array([-0.84997159,  1.63694572, -0.31109291], dtype=float32),
 array([-2.21080685, -0.48928043, -0.26122162], dtype=float32),
 array([-0.84997159,  1.63694572, -0.31109291], dtype=float32)]

```
---
## CMinTable ##

**Scala:**
```scala
val layer = CMinTable()
```
**Python:**
```python
layer = CMinTable()
```

CMinTable takes a bunch of tensors as inputs. These tensors must have
same shape. This layer will merge them by doing an element-wise comparision
and use the min value.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val layer = CMinTable()
layer.forward(T(
  Tensor(T(1.0f, 5.0f, 2.0f)),
  Tensor(T(3.0f, 4.0f, -1.0f)),
  Tensor(T(5.0f, 7.0f, -5.0f))
))
layer.backward(T(
  Tensor(T(1.0f, 5.0f, 2.0f)),
  Tensor(T(3.0f, 4.0f, -1.0f)),
  Tensor(T(5.0f, 7.0f, -5.0f))
), Tensor(T(0.1f, 0.2f, 0.3f)))
```
Gives the output,

```
1.0
4.0
-5.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]

{
  2: 0.0
     0.2
     0.0
     [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
  1: 0.1
     0.0
     0.0
     [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
  3: 0.0
     0.0
     0.3
  [com.intel.analytics.bigdl.tensor.DenseTensor of size 3]
}
```

**Python example:**
```python
from bigdl.nn.layer import CMinTable
import numpy as np

layer = CMinTable()
layer.forward([
  np.array([1.0, 5.0, 2.0]),
  np.array([3.0, 4.0, -1.0]),
  np.array([5.0, 7.0, -5.0])
])

layer.backward([
  np.array([1.0, 5.0, 2.0]),
  np.array([3.0, 4.0, -1.0]),
  np.array([5.0, 7.0, -5.0])
], np.array([0.1, 0.2, 0.3]))

```
Gives the output,

```
array([ 1.,  4., -5.], dtype=float32)

[array([ 0.1, 0., 0.], dtype=float32),
array([ 0., 0.2, 0.], dtype=float32),
array([ 0., 0., 0.30000001], dtype=float32)]

```

