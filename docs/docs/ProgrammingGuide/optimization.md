

## Use Optimizer for Training 

You can use `Optimizer` in BigDL to train a model. 

You need to first create an `Optimizer`, and then call `Optimizer.optimize` to start the training. 

To create an optimizer, you need at least provide model, data, loss function and batch size.

* **model**

A neural network model. May be a layer, a sequence of layers or a
graph of layers.

* **data**

Your training data. As we train models on Spark, one of
the most common distributed data structures is RDD. Of course
you can use DataFrame. Please check the BigDL pipeline example.

The element in the RDD is [Sample](../APIGuide/Data.md#sample), which is actually a sequence of
Tensors. You need to convert your data record(image, audio, text)
to Tensors before you feed them into Optimizer. We also provide
many utilities to do it.

* **loss function**

In supervised machine learning, loss function compares the output of
the model with the ground truth(the labels of the training data). It
outputs a loss value to measure how good the model is(the lower the
better). It also provides a gradient to indicate how to tune the model.

In BigDL, all loss functions are subclass of Criterion. Refer to [Losses](../APIGuide/Losses.md) for a list of defined losses.

* **batch size**

Training is an iterative process. In each iteration, only a batch of data
is used for training the model. You need to specify the batch size. Please note, 
the batch size should be divisible by the total cores number.

Here's an example of how to train a Linear classification model

**scala**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.SparkContext

val conf = Engine.createSparkConf()
    .setAppName("optimization")
    .setMaster("local[*]")
    
val sc = new SparkContext(conf)
Engine.init

// Define the model
val model = Linear[Float](2, 1)
model.bias.zero()

// Generate 2D dummy data, y = 0.1 * x[1] + 0.3 * x[2]
val samples = Seq(
  Sample[Float](Tensor[Float](T(5f, 5f)), Tensor[Float](T(2.0f))),
  Sample[Float](Tensor[Float](T(-5f, -5f)), Tensor[Float](T(-2.0f))),
  Sample[Float](Tensor[Float](T(-2f, 5f)), Tensor[Float](T(1.3f))),
  Sample[Float](Tensor[Float](T(-5f, 2f)), Tensor[Float](T(0.1f))),
  Sample[Float](Tensor[Float](T(5f, -2f)), Tensor[Float](T(-0.1f))),
  Sample[Float](Tensor[Float](T(2f, -5f)), Tensor[Float](T(-1.3f)))
)
val trainData = sc.parallelize(samples, 1)

// Define the model
val optimizer = Optimizer[Float](model, trainData, MSECriterion[Float](), 4)
Engine.init
optimizer.optimize()
println(model.weight)
```

The weight of linear is init randomly. But the output should be like
```
scala> println(model.weight)
0.09316949      0.2887804
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2]
```

**python**
```
from bigdl.nn.layer import Linear
from bigdl.util.common import *
from bigdl.nn.criterion import MSECriterion
from bigdl.optim.optimizer import Optimizer, MaxIteration
import numpy as np

model = Linear(2, 1)
samples = [
  Sample.from_ndarray(np.array([5, 5]), np.array([2.0])),
  Sample.from_ndarray(np.array([-5, -5]), np.array([-2.0])),
  Sample.from_ndarray(np.array([-2, 5]), np.array([1.3])),
  Sample.from_ndarray(np.array([-5, 2]), np.array([0.1])),
  Sample.from_ndarray(np.array([5, -2]), np.array([-0.1])),
  Sample.from_ndarray(np.array([2, -5]), np.array([-1.3]))
]
train_data = sc.parallelize(samples, 1)
init_engine()
optimizer = Optimizer(model, train_data, MSECriterion(), MaxIteration(100), 4)
optimizer.optimize()
model.get_weights()[0]
```

The output should be like
```
array([[ 0.11578175,  0.28315681]], dtype=float32)
```

You can see the model is trained.

### Define when to end the training
You need define when to end the training. It can be several iterations, or how many round
data you want to process, a.k.a epoch.

**scala**
```scala
// The default endWhen in scala is 100 iterations
optimizer.setEndWhen(Trigger.maxEpoch(10))  // Change to 10 epoch
```
You also can use multiple triggers to decide when to end the training:
```scala
// If all of the inner triggers are triggered (logical AND)
optimizer.setEndWhen(
   Trigger.and(Trigger.maxScore(0.99f),Trigger.maxEpoch(10))
)

// If any of the inner triggers are triggered (logical OR)
optimizer.setEndWhen(
   Trigger.or(Trigger.maxScore(0.99f),Trigger.maxEpoch(10))
)

// Inner and/or 
optimizer.setEndWhen(
   Trigger.or(
        Trigger.and( 
            Trigger.maxScore(0.99f), Trigger.maxEpoch(10) 
        ),
        Trigger.maxEpoch(50)
   )
)
```
 

**python**
```
# Python need to define in the constructor
optimizer = Optimizer(model, train_data, MSECriterion(), MaxIteration(100), 4)
```

## Change the optimization algorithm
Gradient based optimization algorithms are the most popular algorithms to train the neural
network model. The most famous one is SGD. SGD has many variants, adagrad, adam, etc.

**scala**
```scala
// The default is SGD
optimizer.setOptimMethod(new Adam())  // Change to adam
```

**python**
```
# Python need to define the optimization algorithm in the constructor
optimizer = Optimizer(model, train_data, MSECriterion(), MaxIteration(100), 4, optim_method = Adam())
```
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

## Validate your model in training
Sometimes, people want to evaluate the model with a separated dataset. When model
performs well on train dataset, but bad on validation dataset, we call the model is overfit or
weak generalization. People may want to evaluate the model every several iterations or 
epochs. BigDL can easily do this by

**scala**
```scala
optimizer.setValidation(trigger, testData, validationMethod, batchSize)
```
**python**
```python
optimizer.set_validation(batch_size, val_rdd, trigger, validationMethod)
```

For validation, you need to provide

* trigger: how often to do validation, maybe each several iterations or epochs
* test data: the separate dataset for test
* validation method: how to evaluate the model, maybe top1 accuracy, etc.
* batch size: how many data evaluate in one time


## Checkpointing

You can configure the optimizer to periodically take snapshots of the model (trained weights, biases, etc.) and optim-method (configurations and states of the optimization) and dump them into files. 

The model snapshot will be named as `model.#iteration_number`, and optim method snapshot will be named as `state.#iteration_number`.

Usage as below.
 
**scala**
```scala
optimizer.setCheckpoint(path, trigger)
```
**python**
```python
optimizer.set_checkpoint(path, trigger,isOverWrite=True)
```
Parameters you need to specify are:

* path - the directory to save the snapshots
* trigger - how often to save the check point 

In scala, you can also use `overWriteCheckpoint()` to enable overwriting any existing snapshot files with the same name (default is disabled). In Python, you can just set parameter isOverWrite (default is True).

**scala**
```scala
optimizer.overWriteCheckpoint()`
```
**python**
```python
optimizer.set_checkpoint(path, trigger,isOverWrite=True)
```
## Resume Training


After training stops, you can resume from any saved point. Choose one of   the model snapshots and the corresponding optim-method snapshot to resume (saved in checkpoint path, details see [Checkpointing](#checkpointing)).     Use `Module.load` (Scala) or `Model.load`(Python) to load the model         snapshot into an model object, and `OptimMethod.load` (Scala and Python) to load optimization method into an OptimMethod  object. Then create a new `Optimizer` with the loaded model and optim       method. Call `Optimizer.optimize`, and you will resume from the point       where the snapshot is taken. Refer to [OptimMethod Load](../APIGuide/Optimizers/OptimMethod.md#load-method) and [Model Load](../APIGuide/Module.md#model-load) for details.

You can also resume training without loading the optim method, if you       intend to change the learning rate schedule or even the optimization        algorithm. Just create an `Optimizer` with loaded model and a new instance  of OptimMethod (both Scala and Python).

## Monitor your training
**scala**
```scala
optimizer.setTrainSummary(trainSummary)
optimizer.setValidationSummary(validationSummary)
```
**python**
```python
set_train_summary(train_summary)
optimizer.set_val_summary(val_summary)
```

See details in [Visualization](visualization.md)

## Performance tuning
For performance investigation, BigDL records the time-consuming distribution on each node for each step(e.g. sync weight, computing).The information can be displayed in the driver log. By default, it is suspended.To turn it on, please follow these steps:

1.Prepare a log4j property file
```
# Root logger option
log4j.rootLogger=INFO, stdout
# Direct log messages to stdout
log4j.appender.stdout=org.apache.log4j.ConsoleAppender
log4j.appender.stdout.Target=System.out
log4j.appender.stdout.layout=org.apache.log4j.PatternLayout
log4j.appender.stdout.layout.ConversionPattern=%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n
log4j.logger.com.intel.analytics.bigdl.optim=DEBUG

```
2.Add an option to your spark-submit command

--conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:where_is_your_log4j_file"




