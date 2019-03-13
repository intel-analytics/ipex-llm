## Optimizer ##

An optimizer is in general to minimize any function with respect to a set of parameters. In case of training a neural network, an optimizer tries to minimize the loss of the neural net with respect to its weights/biases, over the training set.

### Scala API ###

***Factory method***

In summary, you need to supply 3 kinds of paramters to create an optimizer:
1) train data: You could supply 
   a) a sampleRDD and batchSize (with optional featurePadding and labelPadding), or 
   b) a sampleRDD and batchSize and a customized implemenation of trait MiniBatch, or
   c) a DataSet - the type of optimizer created will be dermined by the type of Dataset. 
2) a model
3) a criterion (i.e. loss)
as shown in below interfaces:

```scala
val optimizer = Opimizer[T: ClassTag, D](
      model: Module[T],
      sampleRDD: RDD[Sample[T]],
      criterion: Criterion[T],
      batchSize: Int,
      featurePaddingParam: PaddingParam[T]=null,
      labelPaddingParam: PaddingParam[T]=null)
```
The meaning of parameters are as below:
`model`: model will be optimized.
`sampleRDD`: training Samples.
`criterion`: loss function.
`batchSize`: mini batch size.
`featurePaddingParam`(optional): feature padding strategy.
`labelPaddingParam`(optional): label padding strategy.
 <br>

```scala
val optimizer = Opimizer[T: ClassTag, D](
  model: Module[T],
  dataset: DataSet[D],
  criterion: Criterion[T])
```
`T`: the numeric type(Float/Double).  
`D`: should be a kind of MiniBatch.  
`model`: the model will be optimized.  
`dataset`: the training DataSet.  
`criterion`: the Loss function.
 <br>

```scala
val optimizer = Opimizer[T: ClassTag, D](
      model: Module[T],
      sampleRDD: RDD[Sample[T]],
      criterion: Criterion[T],
      batchSize: Int,
      miniBatch: MiniBatch[T])
```
Apply an optimizer with User-Defined `MiniBatch`.  
`model`: model will be optimized.  
`sampleRDD`: training Samples.  
`criterion`: loss function.  
`batchSize`: mini batch size.  
`miniBatch`: An User-Defined MiniBatch implementation.
 <br>

***Validation***

Function setValidation is to set a validate evaluation in the `optimizer`.
```scala
optimizer.setValidation(
  trigger: Trigger,
  dataset: DataSet[MiniBatch[T]],
  vMethods : Array[ValidationMethod[T])
```
`trigger`: how often to evaluate validation set.  
`dataset`: validate data set in type of DataSet[MiniBatch].  
`vMethods`: a set of ValidationMethod.  
 <br>
```scala
optimizer.setValidation(
  trigger: Trigger,
  sampleRDD: RDD[Sample[T]],
  vMethods: Array[ValidationMethod[T]],
  batchSize: Int)
```
`trigger`: how often to evaluate validation set.  
`sampleRDD`: validate data set in type of RDD[Sample].  
`vMethods`: a set of ValidationMethod.  
`batchSize`: size of mini batch.
 <br>
***Checkpoint***
```scala
optimizer.setCheckpoint(path: String, trigger: Trigger)
```
Function setCheckPoint is used to set a check point saved at `path` triggered by `trigger`.  
`path`: a local/HDFS directory to save checkpoint.  
`trigger`: how often to save the check point.  
 <br>
```scala
val path = optimizer.getCheckpointPath()
```
Function getCheckpointPath is used to get the directory of saving checkpoint.  
 <br>
```scala
optimizer.overWriteCheckpoint()
```
Function overWriteCheckpoint is enable overwrite saving checkpoint.  

***Summary***

```scala
optimizer.setTrainSummary(trainSummary: TrainSummary)
```
Function setTrainSummary is used to enable train summary in this optimizer.  
`trainSummary`: an instance of TrainSummary.  
 <br>
```scala
optimizer.setValidationSummary(validationSummary: ValidationSummary)
```
Function setValidationSummary is used to enable validation summary in this optimizer.  
`validationSummary`: an instance of ValidationSummary.  

***Other important API***
```scala
val trainedModel = optimizer.optimize()
```
Function optimize will start the training.  
 <br>
```scala
optimizer.setModel(newModel: Module[T])
```
Function setModel will set a new model to the optimizer.  
`newModel`: a model will replace the old model in optimizer.  
 <br>
 ```scala
 optimizer.setModelAndOptimMethods(newModel: Module[T], newOptimMethods: Map[String, OptimMethod[T]])
```
Function setModelAndOptimMethods will set a new model and new OptimMethods to the optimizer.  
`newModel`: a model will replace the old model in optimizer.  
`newOptimMethods`: new OptimMethods will replace the old model in optimizer. It's a mapping of submodule -> OptimMethod.
 <br>
```scala

optimizer.setTrainData(sampleRDD: RDD[Sample[T]],
                 batchSize: Int,
                 miniBatch: MiniBatch[T])

optimizer.setTrainData(sampleRDD: RDD[Sample[T]],
                 batchSize: Int,
                 featurePaddingParam: PaddingParam[T]=null,
                 labelPaddingParam: PaddingParam[T])=null

```
the overloaded set of methods `setTrainData` allows user to replace the training data. Each time setTrainData is called, the dataset is replaced and the following call to optimize() will use the new dataset. The meaning of arguments are the same as in the Factory methods:
`sampleRDD`: training Samples.
`batchSize`: mini batch size.
`featurePaddingParam`: feature padding strategy.
`labelPaddingParam`: label padding strategy.
`miniBatch`: A User-Defined MiniBatch implemenation.
 <br>

```scala
optimizer.setCriterion(newCriterion: Criterion[T])
```
`setCriterion` allows user to set a new criterion to replace the old one. 
`newCriterion`: the new Criterion.
 <br>

```scala
optimizer.setState(state: Table)
```
Function setState is used to set a state(learning rate, epochs...) to the `optimizer`.  
`state`: the state to be saved.  
 <br>
```scala
optimizer.setOptimMethod(method : OptimMethod[T])
```
Function setOptimMethod is used to set an optimization method in this `optimizer`.  
`method`: the method to optimize the model in this `optimizer`.  
 <br>
 ```scala
optimizer.setOptimMethods(method: Map[String, OptimMethod[T]])
```
Function setOptimMethods is used to set different optimization methods for submodules in this `optimizer`.  
`method`: the mapping of submodule -> OptimMethod  
 <br>
```scala
optimizer.setEndWhen(endWhen: Trigger)
```
Function setEndWhen is used to declare when to stop the training invoked by `optimize()`.  
`endWhen`: a trigger to stop the training.

### Scala example ###
Here is an example to new an Optimizer with SGD for optimizing LeNet5 model.
```scala
val trainingRDD = ...
val valRDD = ...
val batchSize = 12
// Create an optimizer
val optimizer = Optimizer(
  model = LeNet5(classNum = 10),
  sampleRDD = trainingRDD,
  criterion = ClassNLLCriterion(),
  batchSize = batchSize
).setValidation(Trigger.everyEpoch, valRDD, Array(new Top1Accuracy), batchSize) // set validation method
  .setEndWhen(Trigger.maxEpoch(15)) // set end trigger
  .setOptimMethod(new SGD(learningRate = 0.05)) // set optimize method, Since 0.2.0. Older version should use optimizer.setOptimMethod(new SGD()).setState(T("learningRate" -> 0.05))

val trainedModel = optimizer.optimize()
```

### Python API ###

***Factory method***

```python
optimizer =  Optimizer(model,
                 training_rdd,
                 criterion,
                 end_trigger,
                 batch_size,
                 optim_method=None,
                 bigdl_type="float")
```

`model`: the model will be optimized.  
`training_rdd`: the training dataset.  
`criterion`: the Loss function.  
`end_trigger`: when to end the optimization.  
`batch_size`: size of minibatch.  
`optim_method`:  the algorithm to use for optimization, e.g. SGD, Adagrad, etc. If optim_method is None, the default algorithm is SGD.  
`bigdl_type`: the numeric type(Float/Double).  

***Validation***

Function setValidation is to set a validate evaluation in the `optimizer`.

```python
optimizer.set_validation(batch_size, val_rdd, trigger, val_method=["Top1Accuracy"])
```
`trigger`: how often to evaluation validation set.  
`val_rdd`: validate data set in type of RDD[Sample].  
`val_method`: a list of ValidationMethod, e.g. "Top1Accuracy", "Top5Accuracy", "Loss".  
`batch_size`: size of mini batch.

***Checkpoint***
```python
optimizer.set_checkpoint(checkpoint_trigger,
                      checkpoint_path, isOverWrite=True)
```
Function setCheckPoint is used to set a check point saved at `path` triggered by `trigger`.  
`checkpoint_trigger`: how often to save the check point.
`checkpoint_path`: a local/HDFS directory to save checkpoint.  
`isOverWrite`: whether to overwrite existing snapshots in path.default is True

***Summary***

```python
optimizer.set_train_summary(summary)
```
Set train summary. A TrainSummary object contains information necessary for the optimizer to know how often the logs are recorded, where to store the logs and how to retrieve them, etc. For details, refer to the docs of TrainSummary.
`summary`: an instance of TrainSummary.

```python
optimizer.set_validation_summary(summary)
```
Function setValidationSummary is used to set validation summary. A ValidationSummary object contains information necessary for the optimizer to know how often the logs are recorded, where to store the logs and how to retrieve them, etc. For details, refer to the docs of ValidationSummary.
`summary`: an instance of ValidationSummary.

***Start Training***
```python
trained_model = optimizer.optimize()
```
Function optimize will start the training.

***Set Model***
```python
optimizer.set_model(model)
```
Function setModel will set a new model to the optimizer.  
`model`: a model will replace the old model in optimizer.

***Set Train Data***
```python
optimizer.set_traindata(sample_rdd, batch_size)
```
set_traindata allows user to replace the train data (for optimizer reuse)

***Set Criterion***
```python
optimizer.set_criterion(criterion)
```
set_criterion allows user to replace the criterion (for optimizer reuse)

### Python example ###
Here is an example to new an Optimizer with SGD for optimizing LeNet5 model.
```python
train_data = ...
test_data = ...
batch_size = 12
# Create an Optimizer, Since 0.2.0
optimizer = Optimizer(
  model=lenet_model,
  training_rdd=train_data,
  criterion=ClassNLLCriterion(),
  optim_method=SGD(learningrate=0.01, learningrate_decay=0.0002), # set optim method
  end_trigger=MaxEpoch(15),
  batch_size=batch_size)
  
# Older version, before 0.2.0, use following code: 
# optimizer = Optimizer(
#   model=model,
#   training_rdd=train_data,
#   criterion=ClassNLLCriterion(),
#   optim_method="SGD",
#   state={"learningRate": 0.05},
#   end_trigger=MaxEpoch(training_epochs),
#   batch_size=batch_size)

optimizer.set_validation(
    batch_size=2048,
    val_rdd=test_data,
    trigger=EveryEpoch(),
    val_method=[Top1Accuracy()]
)

# Older version, before 0.2.0, use following code: 
#optimizer.set_validation(
#    batch_size=2048,
#    val_rdd=test_data,
#    trigger=EveryEpoch(),
#    val_method=["Top1Accuracy"]
#)

trained_model = optimizer.optimize()

```

### How BigDL train models in a distributed cluster? ###
BigDL distributed training is data parallelism. The training data is split among workers and cached in memory. A complete model is also cached on each worker. The model only uses the data of the same worker in the training.

BigDL employs a synchronous distributed training. In each iteration, each worker will sync the latest weights, calculate gradients with local data and local model, sync the gradients and update the weights with a given optimization method(e.g. SGD, Adagrad).

In gradients and weights sync, BigDL doesn't use the RDD APIs like(broadcast, reduce, aggregate, treeAggregate). The problem of these methods is every worker needs to communicate with driver, so the driver will become the bottleneck if the parameter is too large or the workers are too many. Instead, BigDL implement a P2P algorithm for parameter sync to remove the bottleneck. For detail of the algorithm, please see the [code](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/optim/DistriOptimizer.scala)
