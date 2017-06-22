## Transpose ##

**Scala:**
```scala
val module = Transpose(permutations)
```
**Python:**
```python
module = Transpose(permutations)
```

Concat is a layer who transpose input along specified dimensions.
permutations are dimension pairs that need to swap.

**Scala example:**
```scala
> val input = Tensor(2, 3).rand()
> val layer = Transpose(Array((1, 2)))

> input
0.6653826	0.25350887	0.33434764	
0.9618287	0.5484164	0.64844745	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

> layer.forward(input)
0.6653826	0.9618287	
0.25350887	0.5484164	
0.33434764	0.64844745	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2]

```

**Python example:**
```python
> layer = Transpose([(1,2)])
> input = np.array([[0.6653826, 0.25350887, 0.33434764], [0.9618287, 0.5484164, 0.64844745]])

> layer.forward(input)
[array([[ 0.66538262,  0.96182871],
       [ 0.25350887,  0.54841638],
       [ 0.33434764,  0.64844745]], dtype=float32)]

```
