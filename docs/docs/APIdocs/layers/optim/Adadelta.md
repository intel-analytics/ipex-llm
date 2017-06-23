## Adadelta ##


*AdaDelta* implementation for *SGD* 
It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`.
http://arxiv.org/abs/1212.5701.

**Scala:**
```scala
Adadelta[Float](decayRate: Double = 0.9, Epsilon: Double = 1e-10)
```
**Python:**
```python
AdaDelta(decayrate = 0.9, epsilon = 1e-10, bigdl_type="float")
```


**Scala example:**
```scala
optimizer.setOptimMethod(new Adadelta[Float](0.9, 1e-10))

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
