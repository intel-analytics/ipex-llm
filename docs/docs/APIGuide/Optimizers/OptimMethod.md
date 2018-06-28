## OptimMethod ##

OptimMethod is used to update model gradient parameters.We have defined SGD method, Adagrad method, etc.
Details about those optim methods, you can refer to [Optim-Methods](Optim-Methods.md).
Now, method construct parameters(e.g."learningRate") and internal training parameters(e.g."epoch") store in optim method instead of state(since version 0.2.0).
Here is mainly to describe how to use those methods when training.
### Set method ###
**scala**
```scala
optimizer.setOptimMethod(method : OptimMethod[T])
```
**python**
```scala
optimizer = Optimizer(
    model,
    training_rdd,
    criterion,
    optim_method,
    end_trigger,
    batch_size)
```
in python, you can set optim method when creating an optimizer.

Sometimes, people want to apply different optimization algorithms for the submodules of the neural network model. 
BigDL provide a method to set optimMethod for submoduels by submodules' name.

**scala**
```scala
val optimMethods = Map("wide" -> new Ftrl[Float](), "deep" -> new Adagrad[Float]())
optimizer.setOptimMethods(optimMethods)
```

**python**
```python
optimMethods = {"wide": Ftrl(), "deep": Adagrad()}
optimizer.setOptimMethods(optimMethods)
```

### Save method ###
```scala
method.save(path: String, overWrite: Boolean = false)
```
`T`: path to save method  
`overWrite`: whether to overwrite or not

When training, you can use optimizer.setCheckPoint(for scala) or optimizer.set_checkpoint(for python) to save methods at regular intervals.

### Load method ###
**scala**
```scala
val method = OptimMethod.load(path : String)
```
`path`: file of optim method path

**python**
```scala
optimizer = OptimMethod.load(path, bigdl_type="float")
```
`bigdl_type`: type of optim method, default is "float"

### Scala example ###
Here is an example to train LeNet5 model with a loading method.
```scala
val trainingRDD = ...
val valRDD = ...
val batchSize = 12
val methodPath = ...
// Load optim method
val method = OptimMethod.load(methodPath)
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
optimizer = Optimizer(
  model=lenet_model,
  training_rdd=train_data,
  criterion=ClassNLLCriterion(),
  optim_method=SGD(learningrate=0.01, learningrate_decay=0.0002), # set optim method
  end_trigger=MaxEpoch(15),
  batch_size=batch_size)
       
optimizer.set_validation(
    batch_size=32,
    val_rdd=test_data,
    trigger=EveryEpoch(),
    val_method=[Top1Accuracy()]
)

optimizer.set_checkpoint(EveryEpoch(), checkpointPath) # set checkpoint to save model and optim method

trained_model = optimizer.optimize()
```
