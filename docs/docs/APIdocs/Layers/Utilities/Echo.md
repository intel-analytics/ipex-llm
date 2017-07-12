## Echo ##

**Scala:**
```scala
val module = Echo()
```
**Python:**
```python
module = Echo()
```

This module is for debug purpose, which can print activation and gradient size in your model topology

**Scala example:**
```scala
val module = Echo()
val input = Tensor(3, 2).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.24058184      0.22737113
0.0028103297    0.18359558
0.80443156      0.07047854
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]

module.forward(input)
res13: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.24058184      0.22737113
0.0028103297    0.18359558
0.80443156      0.07047854
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]
```

**Python example:**
```python
module = Echo()
input = np.random.rand(3,2)
[array([
[ 0.87273163,  0.59974301],
[ 0.09416127,  0.135765  ],
[ 0.11577505,  0.46095625]], dtype=float32)]

module.forward(input)
com.intel.analytics.bigdl.nn.Echo@535c681 : Activation size is 3x2
[array([
[ 0.87273163,  0.59974301],
[ 0.09416127,  0.135765  ],
[ 0.11577505,  0.46095625]], dtype=float32)]

```
