## Adagrad ##

**Scala:**
```scala
val adagrad = new Adagrad(learningRate = 1e-3,
                          learningRateDecay = 0.0,
                          weightDecay = 0.0)

```

 An implementation of Adagrad. See the original paper:
 <http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor._
val adagrad = Adagrad(0.01, 0.0, 0.0)
    def feval(x: Tensor[Float]): (Float, Tensor[Float]) = {
      // (1) compute f(x)
      val d = x.size(1)
      // x1 = x(i)
      val x1 = Tensor[Float](d - 1).copy(x.narrow(1, 1, d - 1))
      // x(i + 1) - x(i)^2
      x1.cmul(x1).mul(-1).add(x.narrow(1, 2, d - 1))
      // 100 * (x(i + 1) - x(i)^2)^2
      x1.cmul(x1).mul(100)
      // x0 = x(i)
      val x0 = Tensor[Float](d - 1).copy(x.narrow(1, 1, d - 1))
      // 1-x(i)
      x0.mul(-1).add(1)
      x0.cmul(x0)
      // 100*(x(i+1) - x(i)^2)^2 + (1-x(i))^2
      x1.add(x0)
      val fout = x1.sum()
      // (2) compute f(x)/dx
      val dxout = Tensor[Float]().resizeAs(x).zero()
      // df(1:D-1) = - 400*x(1:D-1).*(x(2:D)-x(1:D-1).^2) - 2*(1-x(1:D-1));
      x1.copy(x.narrow(1, 1, d - 1))
      x1.cmul(x1).mul(-1).add(x.narrow(1, 2, d - 1)).cmul(x.narrow(1, 1, d - 1)).mul(-400)
      x0.copy(x.narrow(1, 1, d - 1)).mul(-1).add(1).mul(-2)
      x1.add(x0)
      dxout.narrow(1, 1, d - 1).copy(x1)
      // df(2:D) = df(2:D) + 200*(x(2:D)-x(1:D-1).^2);
      x0.copy(x.narrow(1, 1, d - 1))
      x0.cmul(x0).mul(-1).add(x.narrow(1, 2, d - 1)).mul(200)
      dxout.narrow(1, 2, d - 1).add(x0)
      (fout, dxout)
    }
val x = Tensor(2).fill(0)
val config = T("learningRate" -> 1e-1)
for (i <- 1 to 10) {
  adagrad.optimize(feval, x, config, config)
}
x after optimize: 0.27779138
0.07226955
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2]
```

