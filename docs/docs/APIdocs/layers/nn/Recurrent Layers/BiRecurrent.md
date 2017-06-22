## BiRecurrent ##

**Scala:**
```scala
val module = BiRecurrent(merge)
```
**Python:**
```python
module = BiRecurrent(merge)
```

This layer implement a bidirectional recurrent neural network
 * @param merge concat or add the output tensor of the two RNNs. Default is add

**Scala example:**
```scala
val module = BiRecurrent[Double](CAddTable[Double]())
.add(RnnCell[Double](6, 4, Sigmoid[Double]()))
val input = Tensor[Double](Array(1, 2, 6)).rand()
module.forward(input)
(1,.,.) =
1.5163003757543496      0.7957292303334982      1.0589401867459423      0.8726666043439615
1.5356772547825959      0.8603368334520043      1.0032676693770364      0.8880027636229939

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2x4]
```

**Python example:**
```python
module = BiRecurrent(CAddTable()).add(RnnCell(6, 4, Sigmoid()))
input = np.random.rand(1, 2, 6)
module.forward(input)
[array([[
[ 1.38137853,  1.04072285,  0.73663014,  0.70244694],
[ 1.49736834,  0.92239362,  0.89991683,  0.87863046]]], dtype=float32)]
```
