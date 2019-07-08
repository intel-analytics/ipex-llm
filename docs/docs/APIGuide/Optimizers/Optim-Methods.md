## Adam ##

**Scala:**
```scala
val optim = new Adam(learningRate=1e-3, learningRateDecay=0.0, beta1=0.9, beta2=0.999, Epsilon=1e-8)
```
**Python:**
```python
optim = Adam(learningrate=1e-3, learningrate_decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8, bigdl_type="float")
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
## SGD ##

**Scala:**
```scala
val optimMethod = new SGD(learningRate= 1e-3,learningRateDecay=0.0,
                      weightDecay=0.0,momentum=0.0,dampening=Double.MaxValue,
                      nesterov=false,learningRateSchedule=Default(),
                      learningRates=null,weightDecays=null)
```

**Python:**
```python
optim_method = SGD(learningrate=1e-3,learningrate_decay=0.0,weightdecay=0.0,
                   momentum=0.0,dampening=DOUBLEMAX,nesterov=False,
                   leaningrate_schedule=None,learningrates=None,
                   weightdecays=None,bigdl_type="float")
```

A plain implementation of SGD which provides optimize method. After setting 
optimization method when create Optimize, Optimize will call optimization method at the end of 
each iteration.
 
**Scala example:**
```scala
val optimMethod = new SGD[Float](learningRate= 1e-3,learningRateDecay=0.0,
                               weightDecay=0.0,momentum=0.0,dampening=Double.MaxValue,
                               nesterov=false,learningRateSchedule=Default(),
                               learningRates=null,weightDecays=null)
optimizer.setOptimMethod(optimMethod)
```

**Python example:**
```python
optim_method = SGD(learningrate=1e-3,learningrate_decay=0.0,weightdecay=0.0,
                  momentum=0.0,dampening=DOUBLEMAX,nesterov=False,
                  leaningrate_schedule=None,learningrates=None,
                  weightdecays=None,bigdl_type="float")
                  
optimizer = Optimizer(
    model=mlp_model,
    training_rdd=train_data,
    criterion=ClassNLLCriterion(),
    optim_method=optim_method,
    end_trigger=MaxEpoch(20),
    batch_size=32)
```

## Adadelta ##


*AdaDelta* implementation for *SGD* 
It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`.
http://arxiv.org/abs/1212.5701.

**Scala:**
```scala
val optimMethod = new Adadelta(decayRate = 0.9, Epsilon = 1e-10)
```
**Python:**
```python
optim_method = AdaDelta(decayrate = 0.9, epsilon = 1e-10)
```


**Scala example:**
```scala
optimizer.setOptimMethod(new Adadelta(0.9, 1e-10))

```


**Python example:**
```python
optimizer = Optimizer(
    model=mlp_model,
    training_rdd=train_data,
    criterion=ClassNLLCriterion(),
    optim_method=Adadelta(0.9, 0.00001),
    end_trigger=MaxEpoch(20),
    batch_size=32)
```

## RMSprop ##

An implementation of RMSprop (Reference: http://arxiv.org/pdf/1308.0850v5.pdf, Sec 4.2)

* learningRate : learning rate
* learningRateDecay : learning rate decay
* decayRate : decayRate, also called rho
* Epsilon : for numerical stability

## Adamax ##

An implementation of Adamax http://arxiv.org/pdf/1412.6980.pdf

Arguments:

* learningRate : learning rate
* beta1 : first moment coefficient
* beta2 : second moment coefficient
* Epsilon : for numerical stability

Returns:

the new x vector and the function list {fx}, evaluated before the update

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
import com.intel.analytics.bigdl.utils.T

val adagrad = new Adagrad(0.01, 0.0, 0.0)
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

## LBFGS ##

**Scala:**
```scala
val optimMethod = new LBFGS(maxIter=20, maxEval=Double.MaxValue,
                            tolFun=1e-5, tolX=1e-9, nCorrection=100,
                            learningRate=1.0, lineSearch=None, lineSearchOptions=None)
```

**Python:**
```python
optim_method = LBFGS(max_iter=20, max_eval=Double.MaxValue, \
                 tol_fun=1e-5, tol_x=1e-9, n_correction=100, \
                 learning_rate=1.0, line_search=None, line_search_options=None)
```

This implementation of L-BFGS relies on a user-provided line search function
(state.lineSearch). If this function is not provided, then a simple learningRate
is used to produce fixed size steps. Fixed size steps are much less costly than line
searches, and can be useful for stochastic problems.

The learning rate is used even when a line search is provided.This is also useful for
large-scale stochastic problems, where opfunc is a noisy approximation of f(x). In that
case, the learning rate allows a reduction of confidence in the step size.

**Parameters:**

* maxIter - Maximum number of iterations allowed. Default: 20
* maxEval - Maximum number of function evaluations. Default: Double.MaxValue
* tolFun - Termination tolerance on the first-order optimality. Default: 1e-5
* tolX - Termination tol on progress in terms of func/param changes. Default: 1e-9
* learningRate - the learning rate. Default: 1.0
* lineSearch - A line search function. Default: None
* lineSearchOptions - If no line search provided, then a fixed step size is used. Default: None

**Scala example:**
```scala
val optimMethod = new LBFGS(maxIter=20, maxEval=Double.MaxValue,
                            tolFun=1e-5, tolX=1e-9, nCorrection=100,
                            learningRate=1.0, lineSearch=None, lineSearchOptions=None)
optimizer.setOptimMethod(optimMethod)
```

**Python example:**
```python
optim_method = LBFGS(max_iter=20, max_eval=DOUBLEMAX, \
                 tol_fun=1e-5, tol_x=1e-9, n_correction=100, \
                 learning_rate=1.0, line_search=None, line_search_options=None)
                  
optimizer = Optimizer(
    model=mlp_model,
    training_rdd=train_data,
    criterion=ClassNLLCriterion(),
    optim_method=optim_method,
    end_trigger=MaxEpoch(20),
    batch_size=32)
```

## Ftrl ##

**Scala:**
```scala
val optimMethod = new Ftrl(
  learningRate = 1e-3, learningRatePower = -0.5,
  initialAccumulatorValue = 0.1, l1RegularizationStrength = 0.0,
  l2RegularizationStrength = 0.0, l2ShrinkageRegularizationStrength = 0.0)
```

**Python:**
```python
optim_method = Ftrl(learningrate = 1e-3, learningrate_power = -0.5, \
                 initial_accumulator_value = 0.1, l1_regularization_strength = 0.0, \
                 l2_regularization_strength = 0.0, l2_shrinkage_regularization_strength = 0.0)
```

An implementation of (Ftrl)[https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf.]
Support L1 penalty, L2 penalty and shrinkage-type L2 penalty.

**Parameters:**

* learningRate: learning rate
* learningRatePower: double, must be less or equal to zero. Default is -0.5.
* initialAccumulatorValue: double, the starting value for accumulators, require zero or positive values. Default is 0.1.
* l1RegularizationStrength: double, must be greater or equal to zero. Default is zero.
* l2RegularizationStrength: double, must be greater or equal to zero. Default is zero.
* l2ShrinkageRegularizationStrength: double, must be greater or equal to zero. Default is zero. This differs from l2RegularizationStrength above. L2 above is a stabilization penalty, whereas this one is a magnitude penalty.

**Scala example:**
```scala
val optimMethod = new Ftrl(learningRate = 5e-3, learningRatePower = -0.5,
  initialAccumulatorValue = 0.01)
optimizer.setOptimMethod(optimMethod)
```

**Python example:**
```python
optim_method = Ftrl(learningrate = 5e-3, \
    learningrate_power = -0.5, \
    initial_accumulator_value = 0.01)
                  
optimizer = Optimizer(
    model=mlp_model,
    training_rdd=train_data,
    criterion=ClassNLLCriterion(),
    optim_method=optim_method,
    end_trigger=MaxEpoch(20),
    batch_size=32)
```

## ParallelAdam ##
Multi-Thread version of [Adam](#adam).

**Scala:**
```scala
val optim = new ParallelAdam(learningRate=1e-3, learningRateDecay=0.0, beta1=0.9, beta2=0.999, Epsilon=1e-8, parallelNum=Engine.coreNumber())
```
**Python:**
```python
optim = ParallelAdam(learningrate=1e-3, learningrate_decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8, parallel_num=get_node_and_core_number()[1], bigdl_type="float")
```
