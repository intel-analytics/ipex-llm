## LeakyReLU ##

**Scala:**
```scala
layer = LeakyReLU(negval=0.01,inplace=false)
```
**Python:**
```python
layer = LeakyReLU(negval=0.01,inplace=False,bigdl_type="float")
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
val layer = LeakyReLU(negval=0.01,inplace=false)
val input = Tensor(3, 2).rand(-1, 1)
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.6923256      -0.14086828
0.029539397     0.477964
0.5202874       0.10458552
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2]

layer.forward(input)
res7: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.006923256    -0.0014086828
0.029539397     0.477964
0.5202874       0.10458552
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2]
```

**Python example:**
```python
layer = LeakyReLU(negval=0.01,inplace=False,bigdl_type="float")
input = np.random.rand(3, 2)
array([[ 0.19502378,  0.40498206],
       [ 0.97056004,  0.35643192],
       [ 0.25075111,  0.18904582]])

layer.forward(input)
array([[ 0.19502378,  0.40498206],
       [ 0.97056001,  0.35643193],
       [ 0.25075111,  0.18904583]], dtype=float32)
```
