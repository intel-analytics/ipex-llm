---
# **Programming Guide**

---
## **Overview**
Before starting the programming guide, you may have checked out the [Getting Started Page](/Getting-Started) and the [Tutorials page](/Tutorials). This section will introduce the BigDL concepts and APIs for building deep learning applications on Spark.

* [Tensor](#tensor)
* [Table](#table)
* [Module](#module)
    * [Create modules](#create-modules)
    * [Construct complex networks](#construct-complex-networks)
    * [Build neural network models](#build-neural-network-models)
* [Criterion](#criterion)
* [Regularizers](#regularizers)
* [Transformer](#transformer)
* [Sample and MiniBatch](#sample-and-minibatch)
* [Engine](#engine)
* [Optimizer](#optimizer)
    * [How BigDL train models in a distributed cluster](#how-BigDL-train-models-in-a-distributed-cluster?)
* [Validator](#validator)
* [Model Persist](#model-persist)
* [Logging](#logging)
* [Visualization via TensorBoard](#visualization-via-tensorboard)

## **Tensor**
Modeled after the [Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md) class in [Torch](http://torch.ch/), the ```Tensor``` [package](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor) (written in Scala and leveraging [Intel MKL](https://software.intel.com/en-us/intel-mkl)) in BigDL provides numeric computing support for the deep learning applications (e.g., the input, output, weight, bias and gradient of the neural networks).

A ```Tensor``` is essentially a multi-dimensional array of numeric types (e.g., ```Int```, ```Float```, ```Double```, etc.); you may check it out in the interactive Scala shell (by typing ```scala -cp bigdl_0.1-0.1.0-SNAPSHOT-jar-with-dependencies.jar```), for instance:
```scala
scala> import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.Tensor

scala> val tensor = Tensor[Float](2, 3)
tensor: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.0     0.0     0.0
0.0     0.0     0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

## **Table**
Modeled after the [Table](https://github.com/torch/nn/blob/master/doc/table.md) class in [Torch](http://torch.ch/), the ```Table``` class (defined in package ```com.intel.analytics.bigdl.utils```) is widely used in BigDL (e.g., a ```Table``` of ```Tensor``` can be used as the input or output of neural networks). In essence, a ```Table``` can be considered as a key-value map, and there is also a syntax sugar to create a ```Table``` using ```T()``` in BigDL.

```scala
scala> import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.T

scala> T(Tensor[Float](2,2), Tensor[Float](2,2))
res2: com.intel.analytics.bigdl.utils.Table =
 {
        2: 0.0  0.0
           0.0  0.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
        1: 0.0  0.0
           0.0  0.0
           [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
 }

```

## **Module**
Modeled after the [nn](https://github.com/torch/nn) package in [Torch](http://torch.ch/), the ```Module``` class in BigDL represents individual layers of the neural network (such as ```ReLU```, ```Linear```, ```SpatialConvolution```, ```Sequential```, etc.).

### **Create modules**
For instance, we can create a ```Linear``` module as follows:
```scala
scala> import com.intel.analytics.bigdl.numeric.NumericFloat // import global float tensor numeric type
import com.intel.analytics.bigdl.numeric.NumericFloat

scala> import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn._

scala> val f = Linear(3,4) // create the module
mlp: com.intel.analytics.bigdl.nn.Linear[Float] = nn.Linear(3 -> 4)

// let's see what f's parameters were initialized to. ('nn' always inits to something reasonable)
scala> f.weight
res5: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.008662592    0.543819        -0.028795477
-0.30469555     -0.3909278      -0.10871882
0.114964925     0.1411745       0.35646403
-0.16590376     -0.19962183     -0.18782845
[com.intel.analytics.bigdl.tensor.DenseTensor of size 4x3]
```

### **Construct complex networks**
We can use the ```Container``` module (e.g., ```Sequential```, ```Concat```, ```ConcatTable```, etc.) to combine individual models to build complex networks, for instance
```scala
scala> val g = Sum()
g: com.intel.analytics.bigdl.nn.Sum[Float] = nn.Sum

scala> val mlp = Sequential().add(f).add(g)
mlp: com.intel.analytics.bigdl.nn.Sequential[Float] =
nn.Sequential {
  [input -> (1) -> (2) -> output]
  (1): nn.Linear(3 -> 4)
  (2): nn.Sum
}
```

### **Build neural network models**
We can create neural network models, e.g., [LeNet-5](http://yann.lecun.com/exdb/lenet/), using different ```Module``` as follows:

```scala
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn._

object LeNet5 {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Tanh())
      .add(SpatialConvolution(6, 12, 5, 5))
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100))
      .add(Tanh())
      .add(Linear(100, classNum))
      .add(LogSoftMax())
  }
}
```

## **Criterion**
Modeled after the [Criterion](https://github.com/torch/nn/blob/master/doc/criterion.md) class in [Torch](http://torch.ch/), the ```Criterion``` class in BigDL will compute loss and gradient (given prediction and target). See [BigDL Criterions](https://github.com/intel-analytics/BigDL/wiki/Criterion) for a list of supported criterions. 
```scala
scala> val mse = MSECriterion() // mean square error lost, usually used for regression loss
mse: com.intel.analytics.bigdl.nn.MSECriterion[Float] = com.intel.analytics.bigdl.nn.MSECriterion@0

scala> val target = Tensor(3).rand() // create a target tensor randomly
target: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.33631626
0.2535103
0.94784033
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]

scala> val prediction = Tensor(3).rand() // create a predicted tensor randomly
prediction: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.91918194
0.6019384
0.38315287
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3]

scala> mse.forward(prediction, target) // use mse to get the loss, returns 1/n sum_i (yhat_i - t_i)^2
res11: Float = 0.2600022

```

## **Regularizers**
Regularizers allow user to apply penalties to the parameters of layers during the optimization process. The penalties are aggregated to the loss function that the network optimizes.

BigDL provides layer wise and parameter separated regularizers. User can apply different penalties to different layers and even different parameters of the same layer. Hence the exact API will depend on the layers.

### **Example**
```scala
Linear(inputN, outputN, 
       wRegularizer = L2Regularizer(0.1),
       bRegularizer = L2Regularizer(0.1))
```

### **Available penalties**
The defined regularizers are located in `com.intel.analytics.bigdl.optim` package.

Three pre-defined regularizers are available:
```scala
L1L2Regularizer(0.)
L1Regularizer(0.)
L2Regularizer(0.)
```

### **Developing new regularizers**
Users can define their own customized regularizers by inheriting the `Regularizer` trait and overriding the `accRegularization` function. The `accRegularization` function takes two arguments, one is the parameters to be penalized and the other is the gradient of the parameters. The derivatives of the penalty function should be defined in `accRegularization`.

## **Transformer**
Transformer is for pre-processing. In many deep learning workload, input data need to be pre-processed before fed into model. For example, in CNN, the image file need to be decoded from some compressed format(e.g. jpeg) to float arrays, normalized and cropped to some fixed shape. You can also find pre-processing in other types of deep learning work load(e.g. NLP, speech recognition). In BigDL, we provide many pre-process procedures for user. They're implemented as Transformer.

The transformer interface is
```scala
trait Transformer[A, B] extends Serializable {
  def apply(prev: Iterator[A]): Iterator[B]
}
```

It's simple, right? What a transformer do is convert a sequence of objects of Class A to a sequence of objects of Class B.

Transformer is flexible. You can chain them together to do pre-processing. Let's still use the CNN example, say first we need read image files from given paths, then extract the image binaries to array of float, then normalized the image content and crop a fixed size from the image at a random position. Here we need 4 transformers, `PathToImage`, `ImageToArray`, `Normalizor` and `Cropper`. And then chain them together.
```scala
class PathToImage extends Transformer[Path, Image]
class ImageToArray extends Transformer[Image, Array]
class Normalizor extends Transformer[Array, Array]
class Cropper extends Transformer[Array, Array]

PathToImage -> ImageToArray -> Normalizor -> Cropper
```

Another benefit from `Transformer` is code reuse. You may find that for similar tasks, although there's a little difference, many pre-processing steps are same. So instead of a big single pre-process function, break it into small steps can improve the code reuse and save your time.

Transformer can work with Spark easily. For example, to transform RDD[A] to RDD[B]
```scala
val rddA : RDD[A] = ...
val tran : Transformer[A, B] = ...
val rddB : RDD[B] = rdd.mapPartitions(tran(_))
```

Transformer here is different from [Spark ML pipeline Transformer](https://spark.apache.org/docs/latest/ml-pipeline.html). But they serve similar purpose. 

## **Sample and MiniBatch**
**Sample** represent one `item` of your data set. For example, one image in image classification, one word in word2vec and one sentence in RNN language model.

**MiniBatch** represent `a batch of samples`. For computing efficiency, we would like to train/inference data in batches.

You need to convert your data type to Sample or MiniBatch by transformers, and then do optimization or inference. Please note that if you provide Sample format, BigDL will still convert it to MiniBatch automatically before optimization or inference.

## **Engine**
BigDL need some environment variables be set correctly to get a good performance. `Engine.init` method can help you set and verify them.

**How to do in the code?**
```scala
// Scala code example
val conf = Engine.createSparkConf()
val sc = new SparkContext(conf)
Engine.init
```
```python
# Python code example
conf=create_spark_conf()
sc = SparkContext(conf)
init_engine()
```
* If you're in spark-shell, Jupyter notebook or yarn-cluster

As the spark context is pre-created, you need start spark-shell or pyspark with `dist/conf/spark-bigdl.conf` file
```bash
# Spark shell
spark-shell --properties-file dist/conf/spark-bigdl.conf ...
# Jupyter notebook
pyspark --properties-file dist/conf/spark-bigdl.conf ...
```
In your code
```scala
Engine.init    // scala: check spark conf values
```
```python
init_engine()    # python: check spark conf values
```
## **Optimizer**
**Optimizer** represent a optimization process, aka training. 

You need to provide the model, train data set and loss function to start a optimization.
```scala
val optimizer = Optimizer(
  model = model,
  dataset = trainDataSet,
  criterion = new ClassNLLCriterion[Float]()
)
```

You can set other properties of a optimization process. Here's some examples:
* Hyper Parameter
```scala
optimizer.setState(
  T(
    "learningRate" -> 0.01,
    "weightDecay" -> 0.0005,
    "momentum" -> 0.9,
    "dampening" -> 0.0,
    "learningRateSchedule" -> SGD.EpochStep(25, 0.5)
  )
)
```
* Optimization method, the default one is SGD. See [Optimization Algorithms](https://github.com/intel-analytics/BigDL/wiki/Optimization-Algorithms) for a list of supported optimization methods and their usage.
```scala
// Change optimization method to adagrad
optimizer.setOptimMethod(new Adagrad())
```
* When to stop, the default one is stopped after 100 iteration
```scala
// Stop after 10 epoch
optimizer.setEndWhen(Trigger.maxEpoch(10))
```
* Checkpoint
```scala
// Every 50 iteration save current model and training status to ./checkpoint
optimizer.setCheckpoint("./checkpoint", Trigger.severalIteration(50))
```
* Validation
You can provide a separated data set for validation.
```scala
// Every epoch do a validation on valData, use Top1 accuracy metrics
optimizer.setValidation(Trigger.everyEpoch, valData, Array(new Top1Accuracy[Float]))
```

### **How BigDL train models in a distributed cluster?**
BigDL distributed training is data parallelism. The training data is split among workers and cached in memory. A complete model is also cached on each worker. The model only uses the data of the same worker in the training.

BigDL employs a synchronous distributed training. In each iteration, each worker will sync the latest weights, calculate
gradients with local data and local model, sync the gradients and update the weights with a given optimization method(e.g. SGD, Adagrad).

In gradients and weights sync, BigDL doesn't use the RDD APIs like(broadcast, reduce, aggregate, treeAggregate). The problem of these methods is every worker needs to communicate with driver, so the driver will become the bottleneck if the parameter is too large or the workers are too many. Instead, BigDL implement a P2P algorithm for parameter sync to remove the bottleneck. For detail of the algorithm, please see the [code](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/optim/DistriOptimizer.scala)

## **Validator**
Validator represent testing the model with some metrics. The model can be loaded from disk or trained from optimization. The metrics can be Top1 accuracy, Loss, etc. See [Validation Methods](https://github.com/intel-analytics/BigDL/wiki/Validation-Methods) for a list of supported validation methods
```scala
// Test the model with validationSet and Top1 accuracy
val validator = Validator(model, validationSet)
val result = validator.test(Array(new Top1Accuracy[Float]))
```

## **Model Persist**
You can save your model like this
```scala
// Save as Java object
model.save("./model")

// Save as Torch object
model.saveTorch("./model.t7")
```
You can read your model file like this
```scala
// Load from Java object file
Module.load("./model")

// Load from torch file
Module.loadTorch("./model.t7")
```

## **Logging**
In the training, BigDL provide a straight forward logging like this. You can see epoch/iteration/loss/throughput directly from the log.
```
2017-01-10 10:03:55 INFO  DistriOptimizer$:241 - [Epoch 1 0/5000][Iteration 1][Wall Clock XXX] Train 512 in XXXseconds. Throughput is XXX records/second. Loss is XXX.
2017-01-10 10:03:58 INFO  DistriOptimizer$:241 - [Epoch 1 512/5000][Iteration 2][Wall Clock XXX] Train 512 in XXXseconds. Throughput is XXX records/second. Loss is XXX.
2017-01-10 10:04:00 INFO  DistriOptimizer$:241 - [Epoch 1 1024/5000][Iteration 3][Wall Clock XXX] Train 512 in XXXseconds. Throughput is XXX records/second. Loss is XXX.
2017-01-10 10:04:03 INFO  DistriOptimizer$:241 - [Epoch 1 1536/5000][Iteration 4][Wall Clock XXX] Train 512 in XXXseconds. Throughput is XXX records/second. Loss is XXX.
2017-01-10 10:04:05 INFO  DistriOptimizer$:241 - [Epoch 1 2048/5000][Iteration 5][Wall Clock XXX] Train 512 in XXXseconds. Throughput is XXX records/second. Loss is XXX.
```
The DistriOptimizer log level is INFO. Currently, we implement a method named with `redirectSparkInfoLogs` in `spark/utils/LoggerFilter.scala`. You can import and redirect at first.

```scala
import com.intel.analytics.bigdl.utils.LoggerFilter
LoggerFilter.redirectSparkInfoLogs()
```

This method will redirect all logs of `org`, `akka`, `breeze` to `bigdl.log` with `INFO` level, except `org.apache.spark.SparkContext`. And it will output all `ERROR` message in console too.

+ You can disable the redirection with java property `-Dbigdl.utils.LoggerFilter.disable=true`. By default, it will do redirect of all examples and models in our code.
+ You can set where the `bigdl.log` will be generated with `-Dbigdl.utils.LoggerFilter.logFile=<path>`. By default, it will be generated under current workspace.

## **Visualization via TensorBoard**
To enable visualization, you need to [install tensorboard](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/visualization/README.md) first, then `setTrainSummary()` and `setValidationSummary()` to your optimizer before you call `optimize()`:
```scala
val logdir = "mylogdir"
val appName = "myapp"
val trainSummary = TrainSummary(logdir, appName)
val talidationSummary = ValidationSummary(logdir, appName)
optimizer.setTrainSummary(trainSummary)
optimizer.setValidationSummary(validationSummary)
```
After you start to run your spark job, the train and validation log will be saved to "mylogdir/myapp/train" and "mylogdir/myapp/validation". Notice: please change the appName before you start a new job, or the log files will conflict.

As the training started, use command `tensorboard --logdir mylogdir` to start tensorboard. Then open http://[ip]:6006 to watch the training.

* TrainSummary will show "Loss" and "Throughput" each iteration by default. You can use `setSummaryTrigger()` to enable "LearningRate" and "Parameters", or change the "Loss" and "Throughput"'s trigger:
```scala
trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
trainSummary.setSummaryTrigger("Parameters", Trigger.severalIteration(20))
```
Notice: "Parameters" show the histogram of parameters and gradParameters in the model. But getting parameters from workers is a heavy operation, recommend setting the trigger interval to at least 10 iterations. For a better visualization, please give names to the layers in model.

* ValidationSummary will show the result of ValidationMethod set in optimizer.setValidation(), like "Loss" and "Top1Accuracy".

* Summary also provide readScalar function to read scalar summary by tag name. Reading "Loss" from summary:
```scala
val trainLoss = trainSummary.readScalar("Loss")
val validationLoss = validationSummary.readScalar("Loss")
```
