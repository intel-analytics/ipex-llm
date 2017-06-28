## CDivTable ##

**Scala:**
```scala
val module = CDivTable()
```
**Python:**
```python
module = CDivTable()
```

Takes a table with two Tensor and returns the component-wise division between them.

**Scala example:**
```scala
val module = CDivTable[Float]()
val input = T(1 -> Tensor[Float](2,3).rand(), 2 -> Tensor[Float](2,3).rand())
input: com.intel.analytics.bigdl.utils.Table =
 {
        2: 0.802295     0.7113872       0.29395157
           0.6562403    0.06519115      0.20099664
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]
        1: 0.7435388    0.59126955      0.10225375
           0.46819785   0.10572237      0.9861797
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]
 }

module.forward(input)
res6: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.9267648       0.8311501       0.34785917
0.7134549       1.6217289       4.906449
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
module = CDivTable()
input = [np.array([[1, 2, 3],[4, 5, 6]]), np.array([[1, 4, 9],[6, 10, 3]])]
module.forward(input)
[array([
[ 1.,                   0.5     ,    0.33333334],
[ 0.66666669, 0.5       ,  2.        ]], dtype=float32)]
```
