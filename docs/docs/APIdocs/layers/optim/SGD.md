## SGD ##

**Scala:**
```scala
val optimMethod = SGD()
optimMethod.optimize(feval, x, config, state)
```

A plain implementation of SGD which provides optimize method. After setting 
optimization method when create Optimize, Optimize will call optimization method at the end of 
each iteration.
 
**Scala example:**
```scala
val optimMethod = new SGD[Double]
def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
  val r = x.clone()
  r.apply1(2 * _)
  val v = x(Array(1))
  return (v * v, r)
}
val x = Tensor[Double](1)
x.fill(10)
val config = T("learningRate" -> 1e-3)
for (i <- 1 to 10) {
  optimMethod.optimize(feval, x, config, config)
}
println("x after optimize: " + x)
x after optimize: 9.801790433519495
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 1]
```