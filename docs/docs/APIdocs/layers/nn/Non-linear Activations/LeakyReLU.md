## LeakyReLU ##

**Scala:**
```scala
layer = LeakyReLU()
```
**Python:**
```python
layer = LeakyReLU()
```

It is a transfer module that applies LeakyReLU, which parameter
negval sets the slope of the negative part:
 LeakyReLU is defined as:
  f(x) = max(0, x) + negval * min(0, x)

 * @param negval sets the slope of the negative partl, default is 0.01
 * @param inplace if it is true, doing the operation in-place without
                using extra state memory, default is false

**Scala example:**
```scala
val layer = LeakyReLU[Double]()
val input = Tensor[Double](3, 2).rand(-1, 1)
input: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.7173410640098155      -0.885859833098948
-0.700249251909554      0.8654347104020417
-0.2802117452956736     -0.09387178346514702
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x2]

layer.forward(input)
0.7173410640098155      -0.00885859833098948
-0.00700249251909554    0.8654347104020417
-0.002802117452956736   -9.387178346514702E-4
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2]
```

**Python example:**
```python
layer = LeakyReLU()
input = np.random.rand(3, 2)
layer.forward(input)
[array([
[ 0.4860383 ,  0.70988643],
[ 0.85360128,  0.70210862],
[ 0.12464172,  0.90051508]], dtype=float32)]
```
