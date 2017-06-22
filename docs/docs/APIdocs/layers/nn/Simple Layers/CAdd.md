## CAdd ##

**Scala:**
```scala
val module = CAdd(size)
```
**Python:**
```python
module = CAdd(size)
```

This layer has a bias tensor with given size. The bias will be added element wise to the input
tensor. If the element number of the bias tensor match the input tensor, a simply element wise
will be done. Or the bias will be expanded to the same size of the input. The expand means
repeat on unmatched singleton dimension(if some unmatched dimension isn't singleton dimension,
it will report an error). If the input is a batch, a singleton dimension will be add to the first
dimension before the expand.

 * @param size the size of the bias 

**Scala example:**
```scala
val module = CAdd[Double](Array(2, 1))
val input = Tensor[Double](2, 3).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.8605319228954613      0.6784247003961354      0.8780337024945766
0.06954618194140494     0.7573000222910196      0.739172674715519
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2x3]

module.forward(input)
res6: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.6545348415630962      0.4724276190637703      0.6720366211622114
-0.39672035572006964    0.291033484629545       0.2729061370540444
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
module = CAdd([2, 1])
input = np.random.rand(2, 3)
module.forward(input)
[array([
[ 0.68064338,  0.48737401,  0.54009759],
[-0.23572761, -0.58774257, -0.03442413]], dtype=float32)]
```
