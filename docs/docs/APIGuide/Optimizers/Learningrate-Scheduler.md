## Poly ##

**Scala:**
```scala
val lrScheduler = Poly(power=0.5, maxIteration=1000)
```
**Python:**
```python
lr_scheduler = Poly(power=0.5, max_iteration=1000, bigdl_type="float")
```

A learning rate decay policy, where the effective learning rate follows a polynomial decay, to be zero by the max_iteration. Calculation: base_lr (1 - iter/maxIteration) `^` (power)

 `power` coeffient of decay, refer to calculation formula

 `maxIteration` max iteration when lr becomes zero

**Scala example:**
```scala
import com.intel.analytics.bigdl.optim.SGD._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val optimMethod = new SGD[Double](0.1)
optimMethod.learningRateSchedule = Poly(3, 100)
def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
  return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
}
val x = Tensor[Double](Storage(Array(10.0, 10.0)))
optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.1
optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.0970299
```
**Python example:**
```python
optim_method = SGD(0.1)
optimMethod.learningRateSchedule = Poly(3, 100)
```

## Default ##

It is the default learning rate schedule. For each iteration, the learning rate would update with the following formula:
 l_{n + 1} = l / (1 + n * learning_rate_decay) where `l` is the initial learning rate

**Scala:**
```scala
val lrScheduler = Default()
```
**Python:**
```python
lr_scheduler = Default()
```

**Scala example:**
```scala
val optimMethod = new SGD[Double](0.1, 0.1)
def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
  return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
}
val x = Tensor[Double](Storage(Array(10.0, 10.0)))
optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.1
optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.09090909090909091
optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.08333333333333334
```

**Python example:**
```python
optimMethod = SGD(leaningrate_schedule=Default())
```

## NaturalExp ##

A learning rate schedule, which rescale the learning rate by exp ( -decay_rate * iter / decay_step ) referring to tensorflow's learning rate decay # natural_exp_decay

 `decay_step` how often to apply decay

 `gamma` the decay rate. e.g. 0.96

**Scala:**
```scala
val learningRateScheduler = NaturalExp(1, 1)
```

**Scala example:**
```scala
val optimMethod = new SGD[Double](0.1)
optimMethod.learningRateSchedule = NaturalExp(1, 1)
def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
  (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
}
val x = Tensor[Double](Storage(Array(10.0, 10.0)))
val state = T("epoch" -> 0, "evalCounter" -> 0)
optimMethod.state = state
optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.1

optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.036787944117144235

optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.013533528323661271
```

## Exponential ##

A learning rate schedule, which rescale the learning rate by lr_{n + 1} = lr * decayRate `^` (iter / decayStep)

 `decayStep` the inteval for lr decay

 `decayRate` decay rate

 `stairCase` if true, iter / decayStep is an integer division and the decayed learning rate follows a staircase function.

**Scala:**
```scala
val learningRateSchedule = Exponential(10, 0.96)
```

**Python:**
```python
exponential = Exponential(100, 0.1)
```

**Scala example:**
```scala
val optimMethod = new SGD[Double](0.05)
optimMethod.learningRateSchedule = Exponential(10, 0.96)
def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
  (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
}
val x = Tensor[Double](Storage(Array(10.0, 10.0)))
val state = T("epoch" -> 0, "evalCounter" -> 0)
optimMethod.state = state
optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.05

optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.049796306069892535
```

**Python example:**
```python
optimMethod = SGD(leaningrate_schedule=Exponential(100, 0.1))
```

## Plateau ##

Plateau is the learning rate schedule when a metric has stopped improving. Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. It monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.

 `monitor` quantity to be monitored, can be Loss or score

 `factor` factor by which the learning rate will be reduced. new_lr = lr * factor

 `patience` number of epochs with no improvement after which learning rate will be reduced.

 `mode` one of {min, max}. In min mode, lr will be reduced when the quantity monitored has stopped decreasing;
 in max mode it will be reduced when the quantity monitored has stopped increasing

 `epsilon` threshold for measuring the new optimum, to only focus on significant changes.

 `cooldown` number of epochs to wait before resuming normal operation after lr has been reduced.

 `minLr` lower bound on the learning rate.

**Scala:**
```scala
val learningRateSchedule = Plateau(monitor="score", factor=0.1, patience=10, mode="min", epsilon=1e-4f, cooldown=0, minLr=0)
```

**Python:**
```python
plateau = Plateau("score", factor=0.1, patience=10, mode="min", epsilon=1e-4, cooldown=0, minLr=0)
```

**Scala example:**
```scala
val optimMethod = new SGD[Double](0.05)
optimMethod.learningRateSchedule = Plateau(monitor="score", factor=0.1, patience=10, mode="min", epsilon=1e-4f, cooldown=0, minLr=0)
def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
  (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
}
val x = Tensor[Double](Storage(Array(10.0, 10.0)))
val state = T("epoch" -> 0, "evalCounter" -> 0)
optimMethod.state = state
optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)


optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)

```

**Python example:**
```python
optimMethod = SGD(leaningrate_schedule=Plateau("score"))
```

## Warmup ##

A learning rate gradual increase policy, where the effective learning rate increase delta after each iteration. Calculation: base_lr + delta * iteration

 `delta` increase amount after each iteration

**Scala:**
```scala
val learningRateSchedule = Warmup(delta = 0.05)
```

**Python:**
```python
warmup = Warmup(delta=0.05)
```

**Scala example:**
```scala
val lrSchedules = new SequentialSchedule(100)
lrSchedules.add(Warmup(0.3), 3).add(Poly(3, 100), 100)
val optimMethod = new SGD[Double](learningRate = 0.1, learningRateSchedule = lrSchedules)

def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
  return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
}
val x = Tensor[Double](Storage(Array(10.0, 10.0)))
optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.1

optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.4
```

**Python example:**
```python
optimMethod = SGD(leaningrate_schedule=Warmup(0.05))
```

## SequentialSchedule ##

A learning rate scheduler which can stack several learning rate schedulers.

 `iterationPerEpoch` iteration numbers per epoch

**Scala:**
```scala
val learningRateSchedule = SequentialSchedule(iterationPerEpoch=100)
```

**Python:**
```python
sequentialSchedule = SequentialSchedule(iteration_per_epoch=5)
```

**Scala example:**
```scala
val lrSchedules = new SequentialSchedule(100)
lrSchedules.add(Warmup(0.3), 3).add(Poly(3, 100), 100)
val optimMethod = new SGD[Double](learningRate = 0.1, learningRateSchedule = lrSchedules)

def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
  return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
}
val x = Tensor[Double](Storage(Array(10.0, 10.0)))
optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.1

optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.4

optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.7

optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-1.0

optimMethod.optimize(feval, x)
> print(optimMethod.learningRateSchedule.currentRate)
-0.9702989999999999
```

**Python example:**
```python
sequentialSchedule = SequentialSchedule(5)
poly = Poly(0.5, 2)
sequentialSchedule.add(poly, 5)
```

## EpochDecay ##

**Scala:**
```scala
def decay(epoch: Int): Double =
  if (epoch >= 1) 2.0 else if (epoch >= 2) 1.0 else 0.0

val learningRateSchedule = EpochDecay(decay)
```

It is an epoch decay learning rate schedule. The learning rate decays through a function argument on number of run epochs l_{n + 1} = l_{n} * 0.1 `^` decayType(epoch)

 `decayType` is a function with number of run epochs as the argument

**Scala example:**
```scala
def decay(epoch: Int): Double =
  if (epoch == 1) 2.0 else if (epoch == 2) 1.0 else 0.0

val optimMethod = new SGD[Double](1000)
optimMethod.learningRateSchedule = EpochDecay(decay)
def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
  return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
}
val x = Tensor[Double](Storage(Array(10.0, 10.0)))
val state = T("epoch" -> 0)
for(e <- 1 to 3) {
  state("epoch") = e
  optimMethod.state = state
  optimMethod.optimize(feval, x)
  if(e <= 1) {
    assert(optimMethod.learningRateSchedule.currentRate==10)
  } else if (e <= 2) {
    assert(optimMethod.learningRateSchedule.currentRate==100)
  } else {
    assert(optimMethod.learningRateSchedule.currentRate==1000)
  }
}
```

## Regime ##

A structure to specify hyper parameters by start epoch and end epoch. Usually work with [[EpochSchedule]].

 `startEpoch` start epoch

 `endEpoch` end epoch

 `config` config table contains hyper parameters

## EpochSchedule ##

A learning rate schedule which configure the learning rate according to some pre-defined [[Regime]]. If the running epoch is within the interval of a regime `r` [r.startEpoch, r.endEpoch], then the learning
 rate will take the "learningRate" in r.config.

 `regimes` an array of pre-defined [[Regime]].

**Scala:**
```scala
val regimes: Array[Regime] = Array(
  Regime(1, 3, T("learningRate" -> 1e-2, "weightDecay" -> 2e-4)),
  Regime(4, 7, T("learningRate" -> 5e-3, "weightDecay" -> 2e-4)),
  Regime(8, 10, T("learningRate" -> 1e-3, "weightDecay" -> 0.0))
)
val learningRateScheduler = EpochSchedule(regimes)
```

**Scala example:**
```scala
val regimes: Array[Regime] = Array(
  Regime(1, 3, T("learningRate" -> 1e-2, "weightDecay" -> 2e-4)),
  Regime(4, 7, T("learningRate" -> 5e-3, "weightDecay" -> 2e-4)),
  Regime(8, 10, T("learningRate" -> 1e-3, "weightDecay" -> 0.0))
)

val state = T("epoch" -> 0)
val optimMethod = new SGD[Double](0.1)
optimMethod.learningRateSchedule = EpochSchedule(regimes)
def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
  return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
}
val x = Tensor[Double](Storage(Array(10.0, 10.0)))
for(e <- 1 to 10) {
  state("epoch") = e
  optimMethod.state = state
  optimMethod.optimize(feval, x)
  if(e <= 3) {
    assert(optimMethod.learningRateSchedule.currentRate==-1e-2)
    assert(optimMethod.weightDecay==2e-4)
  } else if (e <= 7) {
    assert(optimMethod.learningRateSchedule.currentRate==-5e-3)
    assert(optimMethod.weightDecay==2e-4)
  } else if (e <= 10) {
    assert(optimMethod.learningRateSchedule.currentRate==-1e-3)
    assert(optimMethod.weightDecay==0.0)
  }
}
```

## EpochStep ##

A learning rate schedule which rescale the learning rate by `gamma` for each `stepSize` epochs.

 `stepSize` For how many epochs to update the learning rate once

 `gamma` the rescale factor

 **Scala:**
 ```scala
 val learningRateScheduler = EpochStep(1, 0.5)
 ```

 **Scala example:**
 ```scala
 val optimMethod = new SGD[Double](0.1)
 optimMethod.learningRateSchedule = EpochStep(1, 0.5)
 def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
   (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
 }
 val x = Tensor[Double](Storage(Array(10.0, 10.0)))
 val state = T("epoch" -> 0)
 for(e <- 1 to 10) {
   state("epoch") = e
   optimMethod.state = state
   optimMethod.optimize(feval, x)
   assert(optimMethod.learningRateSchedule.currentRate==(-0.1 * Math.pow(0.5, e)))
 }
 ```
