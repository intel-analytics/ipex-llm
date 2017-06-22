## CosineDistance ##

**Scala:**
```scala
val module = CosineDistance()
```
**Python:**
```python
module = CosineDistance()
```

CosineDistance creates a module that takes a table of two vectors (or matrices if in batch mode) as input and outputs the cosine distance between them.

**Scala example:**
```scala
val module = CosineDistance()
val t1 = Tensor(3).randn()
val t2 = Tensor(3).randn()
val input = T(t1, t2)

> println(input)
{
	2: -1.178431998847704
	   -0.835195590415135
	   1.4361724069581836
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]
	1: 0.1229851310257113
	   2.081862365910659
	   0.33761222385799744
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]
 }

> module.forward(input)
-0.325067095423888
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1]
```

**Python example:**
```python
module = CosineDistance()
t1 = np.random.randn(3)
t2 = np.random.randn(3)
input = [t1, t2]

> print input
[array([-1.77578888,  1.22056291,  0.19355876]), array([-1.9792508 , -1.46121938,  0.40647557])]

> module.forward(input)
[array([ 0.33549139], dtype=float32)]
```
