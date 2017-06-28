## SGD ##

**Scala:**
```scala
val optimMethod = SGD(learningRate= 1e-3,learningRateDecay=0.0,
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
