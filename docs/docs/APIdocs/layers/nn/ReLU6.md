## ReLU6 ##

**Scala:**
```scala
val module = ReLU6(inplace: Boolean = false)
```
**Python:**
```python
module = ReLU6(inplace=False)
```

Same as ReLU except that the rectifying function f(x) saturates at x = 6 
ReLU6 is defined as:
`f(x) = min(max(0, x), 6)`

**Scala example:**
```scala
val module = ReLU6()

println(module.forward(Tensor.range(-2, 8, 1)))
```
Output is
```
com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.0
0.0
0.0
1.0
2.0
3.0
4.0
5.0
6.0
6.0
6.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 11]
```

**Python example:**
```python
module = ReLU6()

print(module.forward(np.arange(-2, 9, 1)))
```
Output is
```
[array([ 0.,  0.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  6.,  6.], dtype=float32)]
```
