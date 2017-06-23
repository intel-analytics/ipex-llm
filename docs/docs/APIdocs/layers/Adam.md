## Adam ##

**Scala:**
```scala
val optim = Adam(learningRate, learningRateDecay, beta1, beta2, Epsilon)
```
**Python:**
```python
optim = Adam(learningRate, learningRateDecay, beta1, beta2, Epsilon)
```

An implementation of Adam optimization, first-order gradient-based optimization of stochastic  objective  functions. http://arxiv.org/pdf/1412.6980.pdf

 `learningRate` learning rate. Default value is 1e-3. 
 
 `learningRateDecay` learning rate decay. Default value is 0.0.
 
 `beta1` first moment coefficient. Default value is 0.9.
 
 `beta2` second moment coefficient. Default value is 0.999.
 
 `Epsilon` for numerical stability. Default value is 1e-8.
 

**Scala example:**
```scala
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

val optm = new Adam[Double]
def rosenBrock(x: Tensor[Double]): (Double, Tensor[Double]) = {
    // (1) compute f(x)
    val d = x.size(1)

    // x1 = x(i)
    val x1 = Tensor[Double](d - 1).copy(x.narrow(1, 1, d - 1))
    // x(i + 1) - x(i)^2
    x1.cmul(x1).mul(-1).add(x.narrow(1, 2, d - 1))
    // 100 * (x(i + 1) - x(i)^2)^2
    x1.cmul(x1).mul(100)

    // x0 = x(i)
    val x0 = Tensor[Double](d - 1).copy(x.narrow(1, 1, d - 1))
    // 1-x(i)
    x0.mul(-1).add(1)
    x0.cmul(x0)
    // 100*(x(i+1) - x(i)^2)^2 + (1-x(i))^2
    x1.add(x0)

    val fout = x1.sum()

    // (2) compute f(x)/dx
    val dxout = Tensor[Double]().resizeAs(x).zero()
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
val x = Tensor[Double](2).fill(0)
val config = T("learningRate" -> 0.002)
> print(optm.optimize(rosenBrock, x, config))
(0.001999999683772284
0.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2],[D@302d88d8)
```
