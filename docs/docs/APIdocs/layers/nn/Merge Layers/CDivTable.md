## CDivTable ##

**Scala:**
```scala
val module = CDivTable()
```
**Python:**
```python
module = CDivTable()
```

Description
Takes a table with two Tensor and returns the component-wise division between them.

**Scala example:**
```scala
val module = CDivTable[Double]()
val input = T(1 -> Tensor[Double](2,3).rand(), 2 -> Tensor[Double](2,3).rand())
input: com.intel.analytics.bigdl.utils.Table =
 {
        2: 0.6415986611973494   0.02757258014753461     0.8064426234923303
           0.7248881512787193   0.7607566525693983      0.2164502516388893
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2x3]
        1: 0.4365965509787202   0.8974416423588991      0.7740533400792629
           0.6623019925318658   0.2667063446715474      0.22057452122680843
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2x3]
 }

module.forward(input)
res1: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.6804823285696155      32.54833742641758       0.9598368408743024
0.9136609439174167      0.35058036465506653     1.0190541223985259
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
