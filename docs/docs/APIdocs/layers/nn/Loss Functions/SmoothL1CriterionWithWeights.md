## SmoothL1CriterionWithWeights ##

**Scala:**
```scala
val smcod = SmoothL1CriterionWithWeights[Float](sigma, num)
```
**Python:**
```python
smcod = SmoothL1CriterionWithWeights(sigma, num)
```

a smooth version of the AbsCriterion
It uses a squared term if the absolute element-wise error falls below 1.
It is less sensitive to outliers than the MSECriterion and in some cases
prevents exploding gradients (e.g. see "Fast R-CNN" paper by Ross Girshick).

```
   d = (x - y) * w_in
  
  loss(x, y, w_in, w_out)
              | 0.5 * (sigma * d_i)^2 * w_out          if |d_i| < 1 / sigma / sigma
   = 1/n \sum |
              | (|d_i| - 0.5 / sigma / sigma) * w_out   otherwise
```

**Scala example:**
```scala
val smcod = SmoothL1CriterionWithWeights[Float](2.4f, 2)

val inputArr = Array(1.1, -0.8, 0.1, 0.4, 1.3, 0.2, 0.2, 0.03)
val targetArr = Array(0.9, 1.5, -0.08, -1.68, -0.68, -1.17, -0.92, 1.58)
val inWArr = Array(-0.1, 1.7, -0.8, -1.9, 1.0, 1.4, 0.8, 0.8)
val outWArr = Array(-1.9, -0.5, 1.9, -1.0, -0.2, 0.1, 0.3, 1.1)

val input = Tensor(Storage(inputArr.map(x => x.toFloat)))
val target = T()
target.insert(Tensor(Storage(targetArr.map(x => x.toFloat))))
target.insert(Tensor(Storage(inWArr.map(x => x.toFloat))))
target.insert(Tensor(Storage(outWArr.map(x => x.toFloat))))
 
val output = smcod.forward(input, target)
val gradInput = smcod.backward(input, target)

> println(output)
  output: Float = -2.17488
> println(gradInput)
-0.010944003
0.425
0.63037443
-0.95
-0.1
0.07
0.120000005
-0.44000003
[com.intel.analytics.bigdl.tensor.DenseTensor of size 8]
```

**Python example:**
```python
smcod = SmoothL1CriterionWithWeights(2.4, 2)

input = np.array([1.1, -0.8, 0.1, 0.4, 1.3, 0.2, 0.2, 0.03]).astype("float32")
targetArr = np.array([0.9, 1.5, -0.08, -1.68, -0.68, -1.17, -0.92, 1.58]).astype("float32")
inWArr = np.array([-0.1, 1.7, -0.8, -1.9, 1.0, 1.4, 0.8, 0.8]).astype("float32")
outWArr = np.array([-1.9, -0.5, 1.9, -1.0, -0.2, 0.1, 0.3, 1.1]).astype("float32")
target = [targetArr, inWArr, outWArr]

output = smcod.forward(input, target)
gradInput = smcod.backward(input, target)

> output
-2.17488
> gradInput
[array([-0.010944  ,  0.42500001,  0.63037443, -0.94999999, -0.1       ,
         0.07      ,  0.12      , -0.44000003], dtype=float32)]
```
