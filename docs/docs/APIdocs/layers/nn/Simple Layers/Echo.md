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
val module = Echo[Double]()
val input = Tensor[Double](3, 2).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.9283763968851417      0.29231453547254205
0.5617080931551754      0.2624172316864133
0.3538190172985196      0.6381787075661123
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x2]

module.forward(input)
com.intel.analytics.bigdl.nn.Echo@c67b4550 : Activation size is 3x2
res1: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.9283763968851417      0.29231453547254205
0.5617080931551754      0.2624172316864133
0.3538190172985196      0.6381787075661123
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x2]
```

**Python example:**
```python
module = Echo()
input = np.random.rand(3,2)
module.forward(input)
com.intel.analytics.bigdl.nn.Echo@535c681 : Activation size is 3x2
[array([
[ 0.87273163,  0.59974301],
[ 0.09416127,  0.135765  ],
[ 0.11577505,  0.46095625]], dtype=float32)]

```
