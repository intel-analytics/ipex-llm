## Usage of optimizers

An optimizer is one of the two arguments required for compiling a model.

**Scala:**

```scala
model.compile(loss = "mean_squared_error", optimizer = "sgd")
```

**Python:**

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

**Scala:**

```scala
model.compile(loss = "mean_squared_error", optimizer = Adam())
```

**Python:**

```python
model.compile(loss='mean_squared_error', optimizer=Adam())
```

---

## Available optimizers

## SGD

A plain implementation of SGD which provides optimize method. After setting 
optimization method when create Optimize, Optimize will call optimization method at the end of 
each iteration.

**Scala:**
```scala
val optimMethod = SGD(learningRate = 1e-3, learningRateDecay = 0.0, 
                      weightDecay = 0.0, momentum = 0.0, dampening = Double.MaxValue, 
                      nesterov = false, learningRateSchedule = Default(), 
                      learningRates = null, weightDecays = null)
```

Parameters:

* `learningRate` : learning rate
* `learningRateDecay` : learning rate decay
* `weightDecay` : weight decay
* `momentum` : momentum
* `dampening` : dampening for momentum
* `nesterov` : enables Nesterov momentum
* `learningRateSchedule` : learning rate scheduler
* `learningRates` : 1D tensor of individual learning rates
* `weightDecays` : 1D tensor of individual weight decays

**Python:**
```python 
optim_method = SGD(learningrate=1e-3, learningrate_decay=0.0, weightdecay=0.0, 
                   momentum=0.0, dampening=DOUBLEMAX, nesterov=False, 
                   leaningrate_schedule=None, learningrates=None, 
                   weightdecays=None)
```

Parameters:

* `learningrate` : learning rate
* `learningrate_decay` : learning rate decay
* `weightdecay` : weight decay
* `momentum` : momentum
* `dampening` : dampening for momentum
* `nesterov` : enables Nesterov momentum
* `leaningrate_schedule` : learning rate scheduler
* `learningrates` : 1D tensor of individual learning rates
* `weightdecays` : 1D tensor of individual weight decays

## Adam

An implementation of Adam optimization, first-order gradient-based optimization of stochastic  objective  functions. <http://arxiv.org/pdf/1412.6980.pdf>

**Scala:**
```scala
val optimMethod = new Adam(learningRate = 1e-3, learningRateDecay = 0.0, beta1 = 0.9, beta2 = 0.999, Epsilon = 1e-8)
```

Parameters:

* `learningRate` learning rate. Default value is 1e-3. 
* `learningRateDecay` learning rate decay. Default value is 0.0.
* `beta1` first moment coefficient. Default value is 0.9.
* `beta2` second moment coefficient. Default value is 0.999.
* `Epsilon` for numerical stability. Default value is 1e-8.

**Python:**
```python
optim_method = Adam(learningrate=1e-3, learningrate_decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8)
```

Parameters:

* `learningrate` learning rate. Default value is 1e-3. 
* `learningrate_decay` learning rate decay. Default value is 0.0.
* `beta1` first moment coefficient. Default value is 0.9.
* `beta2` second moment coefficient. Default value is 0.999.
* `epsilon ` for numerical stability. Default value is 1e-8.

## Adamax

An implementation of Adamax: <http://arxiv.org/pdf/1412.6980.pdf>

**Scala:**
```scala
val optimMethod = new Adamax(learningRate = 0.002, beta1 = 0.9, beta2 = 0.999, Epsilon = 1e-8)
```

Parameters:

* `learningRate` : learning rate
* `beta1` : first moment coefficient
* `beta2` : second moment coefficient
* `Epsilon` : for numerical stability

**Python:**
```python
optim_method = Adam(learningrate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8)
```

Parameters:

* `learningrate` : learning rate
* `beta1` : first moment coefficient
* `beta2` : second moment coefficient
* `epsilon` : for numerical stability

## Adadelta

*AdaDelta* implementation for *SGD* 
It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`.
<http://arxiv.org/abs/1212.5701.>

**Scala:**
```scala
val optimMethod = Adadelta(decayRate = 0.9, Epsilon = 1e-10)
```

Parameters:

* `decayRate` : decayRate, also called interpolation parameter rho
* `Epsilon` : for numerical stability

**Python:**
```python
optim_method = AdaDelta(decayrate=0.9, epsilon=1e-10)
```

Parameters:

* `decayrate` : decayRate, also called interpolation parameter rho
* `epsilon` : for numerical stability

## Adagrad

 An implementation of Adagrad. See the original paper:
 <http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>

**Scala:**
```scala
val optimMethod = new Adagrad(learningRate = 1e-3, learningRateDecay = 0.0, weightDecay = 0.0)
```

* `learningRate` : learning rate
* `learningRateDecay` : learning rate decay
* `weightDecay` : weight decay

**Python:**
```python
optim_method = Adagrad(learningrate=1e-3, learningrate_decay=0.0, weightdecay=0.0)
```

Parameters:

* `learningrate` : learning rate
* `learningrate_decay` : learning rate decay
* `weightdecay` : weight decay

## Rmsprop

An implementation of RMSprop (Reference: <http://arxiv.org/pdf/1308.0850v5.pdf>, Sec 4.2)

**Scala:**
```scala
val optimMethod = new RMSprop(learningRate = 0.002, learningRateDecay = 0.0, decayRate = 0.99, Epsilon = 1e-8)
```

Parameters:

* `learningRate` : learning rate
* `learningRateDecay` : learning rate decay
* `decayRate` : decayRate, also called rho
* `Epsilon` : for numerical stability

**Python:**
```python
optim_method = RMSprop(learningrate=0.002, learningrate_decay=0.0, decayrate=0.99, epsilon=1e-8)
```

Parameters:

* `learningrate` : learning rate
* `learningrate_decay` : learning rate decay
* `decayrate` : decayRate, also called rho
* `epsilon` : for numerical stability
