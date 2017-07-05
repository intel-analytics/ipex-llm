## Adam ##

**Scala:**
```scala
val optim = new Adam(learningRate=1e-3, learningRateDecay=0.0, beta1=0.9, beta2=0.999, Epsilon=1e-8)
```
**Python:**
```python
optim = Adam(learningRate=1e-3, learningRateDecay-0.0, beta1=0.9, beta2=0.999, Epsilon=1e-8, bigdl_type="float")
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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val optm = new Adam(learningRate=0.002)
def rosenBrock(x: Tensor[Float]): (Float, Tensor[Float]) = {
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
> print(optm.optimize(rosenBrock, x))
(0.0019999996
0.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 2],[D@302d88d8)
```
**Python example:**
```python
optim_method = Adam(learningrate=0.002)
                  
optimizer = Optimizer(
    model=mlp_model,
    training_rdd=train_data,
    criterion=ClassNLLCriterion(),
    optim_method=optim_method,
    end_trigger=MaxEpoch(20),
    batch_size=32)

```