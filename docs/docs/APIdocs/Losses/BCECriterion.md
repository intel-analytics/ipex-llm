## BCECriterion ##

**Scala:**
```scala
val criterion = BCECriterion[Float]()
```
**Python:**
```python
criterion = BCECriterion()
```

 This loss function measures the Binary Cross Entropy between the target and the output
``` 
 loss(o, t) = - 1/n sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
```
 or in the case of the weights argument being specified:
```
 loss(o, t) = - 1/n sum_i weights[i] * (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
```
 By default, the losses are averaged for each mini-batch over observations as well as over
 dimensions. However, if the field sizeAverage is set to false, the losses are instead summed.

**Scala example:**
```scala

val criterion = BCECriterion[Float]()
val input = Tensor[Float](3, 1).rand

val target = Tensor[Float](3)
target(1) = 1
target(2) = 0
target(3) = 1

val output = criterion.forward(input, target)
val gradInput = criterion.backward(input, target)

> println(target)
res25: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0
0.0
1.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3]

> println(output)
output: Float = 0.9009579

> println(gradInput)
gradInput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-1.5277504
1.0736246
-0.336957
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x1]

```

**Python example:**
```python

criterion = BCECriterion()
input = np.random.uniform(0, 1, (3, 1)).astype("float32")
target = np.array([1, 0, 1])
output = criterion.forward(input, target)
gradInput = criterion.backward(input, target)

> output
1.9218739
> gradInput
[array([[-4.3074522 ],
        [ 2.24244714],
        [-1.22368968]], dtype=float32)]

```
