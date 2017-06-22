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
val module = Pack(2)
val input1 = Tensor(2, 2).randn()
val input2 = Tensor(2, 2).randn()
val input = T()
input(1) = input1
input(2) = input2

> input
 {
	2: -0.6625016209189003	2.703488979320708
	   -0.09039655154427263	1.5350540102993306
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2x2]
	1: -1.2156987233048036	-1.9014829828283455
	   -1.549556299369814	0.49998143405510703
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2x2]
 }
 
> module.forward(input)
(1,.,.) =
-1.2156987233048036	-1.9014829828283455
-0.6625016209189003	2.703488979320708

(2,.,.) =
-1.549556299369814	0.49998143405510703
-0.09039655154427263	1.5350540102993306

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2]

```

**Python example:**
```python
module = Pack(2)
input1 = np.random.randn(2, 2)
input2 = np.random.randn(2, 2)
input = [input1, input2]

> print input
[array([[ 0.92741416, -3.29826586],
       [-0.03147819, -0.10049306]]), array([[-0.27146461, -0.25729802],
       [ 0.1316149 ,  1.27620145]])]
       
> module.forward(input)
array([[[ 0.92741418, -3.29826593],
        [-0.27146462, -0.25729802]],

       [[-0.03147819, -0.10049306],
        [ 0.13161489,  1.27620149]]], dtype=float32)
```
