## OptimMethod ##

OptimMethod is used to update model gradient parameters.We have defined SGD method, Adagrad method, etc.
Details about those optim methods, you can refer to [Optim-Methods](Optim-Methods.md).
Now, method parameters(e.g."learningRate") and internal training parameters(e.g."epoch") store in Table state.
Here is mainly to describe how to use those methods when training
### Set method ###
**scala**
```scala
optimizer.setOptimMethod(method : OptimMethod[T])
optimizer.setState(state : Table) // set method parameters
```
**python**
```scala
optimizer = Optimizer(
    model,
    training_rdd,
    criterion,
    optim_method,
    state,
    end_trigger,
    batch_size)
```
in python, you can set optim method when creating an optimizer

### Save method parameters ###
In this release, we can just support save parameters of optim method, but no method name.
When training, you can use optimizer.setCheckPoint(for scala) or optimizer.set_checkpoint(for python) to save parameters at regular intervals.

### Load method parameters ###
Method parameters are stored in state, so you can load state like this:
```scala
val state = T.load(path : String)
```
`path`: file of state path

Python can't support loading optim method from a snapshot in this release.

### Scala example ###
Here is an example to train LeNet5 model with a loading state.
```scala
val trainingRDD = ...
val valRDD = ...
val batchSize = 12
val methodPath = ...
// Load optim method
val state = T.load(state)
// Create an optimizer
val optimizer = Optimizer(
  model = LeNet5(classNum = 10),
  sampleRDD = trainingRDD,
  criterion = ClassNLLCriterion(),
  batchSize = batchSize
).setValidation(Trigger.everyEpoch, valRDD, Array(new Top1Accuracy), batchSize)
  .setEndWhen(Trigger.maxEpoch(15))

optimizer.setOptimMethod(method) // set optim method

optimizer.setCheckpoint(param.checkpoint.get, checkpointTrigger) // set checkpoint to save model and optim method

val trainedModel = optimizer.optimize()
```

### Python example ###
Here is an example to train LeNet5 model with SGD method.
```python
train_data = ...
test_data = ...
batch_size = 12
state = {"learningRate": 0.01,
         "learningRateDecay": 0.0002}
optimizer = Optimizer(
  model=lenet_model,
  training_rdd=train_data,
  criterion=ClassNLLCriterion(),
  optim_method="SGD",
  state=state
  end_trigger=MaxEpoch(15),
  batch_size=batch_size)
       
optimizer.set_validation(
    batch_size=32,
    val_rdd=test_data,
    trigger=EveryEpoch(),
    val_method=["Top1Accuracy"]
)

optimizer.set_checkpoint(EveryEpoch(), checkpointPath) # set checkpoint to save model and optim method

trained_model = optimizer.optimize()
```
