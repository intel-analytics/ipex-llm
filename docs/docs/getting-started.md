---
## Before using BigDL

To use BigDL *Scala* API, the first thing is to obtain the BigDL libraries. You can either download the pre-built BigDL libs, or build the libs from source code (available at [BigDL Github](https://github.com/intel-analytics/BigDL)). Then in your program, you need to ensure the SparkContext is created successfully and initialze BigDL engine. Refer to [Install](UserGuide/install.md) and [Run](UserGuide/run.md) for details about how to build BigDL and run a scala program.  

To use BigDL *Python* API, you can install BigDL python simply via pip. You may use Python API in an interactive shell, or run a program in commandline, or use Jupyter notebooks. Before calling BigDL API's in your program, you have to ensure the SparkContext is succesfully created and initialize BigDL engine. Refer to [Python Install](PythonSupport/python-install.md) and [Python Run](PythonSupport/python-run.md) for details about how to install python and run python programs.

---

## Prepare your Data

You data need to be transformed into specific data structures in order to be fed into BigDL for training, evaluation and prediction.
 
Below are several data structures that you need to know when using BigDL. 

* ```Tensor``` is a multi-dimensional array of basic numeric types (e.g., ```Int```, ```Float```,       ```Double```, etc.) It is the most essential data structure in BigDL, composing the basic data flow inside the nerual network (i.e. the input, output, weight, bias and gradient of many layers). Refer to [Tensor API doc](APIdocs/Data/merged-Data#tensor) for details about the numeric computation functions provided in BigDL. 
* `Table` is key-value map. Some of the layers accepts Table as input or output a Table (e.g. [ConcatTable](APIdocs/Layers/Containers/merged-Containers/#concattable)). Find detailed usage in [Table API doc](APIdocs/Data/merged-Data#table). There is a syntax sugar to to using ```T()``` to create Tables in BigDL.
 
* `Sample` is a **(feature, label)** pair. A `Sample` can be created from two `Tensors` (in Scala) or two `numpy arrays` (in Python). Refer to [Sample API doc](APIdocs/Data/merged-Data#sample) for detailed usage.

[TODO need elaboration on `Dataset`(scala only)? `MiniBatch` seems rarely used by user, so skip it here]

You need to convert your dataset into `RDD` of `Samples`, and then feed your data into Optimizer for training, validation or prediction. Refer to [Optimizer docs](APIdocs/Optimizers/DistriOptimizer.md) for details.

[TODO We have provided several preprocessing utilities to help you transform your data.]

---

## Use BigDL for Prediction only

If you have an existing model and want to use BigDL only for prediction, you need to first load the model, and then do prediction or evaluation. 

BigDL supports loading models trained and saved in BigDL, or a trained Caffe model, or a trained Tensorflow model. 

* To load a BigDL model, you can use `module.load` interface. Refer to [Module API docs](APIdocs/Model/ModuleAPI.md) for details.  
* To load a Tensorflow model, refer to [Tensorflow Support](PythonSupport/tensorflow-support.md) for details.
* [TODO To load a Caffe model, refer to ...]

Once you have a loaded model, you can call `model.predict()` to do predictions. Note that prediction need your data to be prepared into designated format. For how to prepare your data, refer to section [Prepare your Data](#prepare-your-data). 

If you are using predicted result as a component inside a Spark ML pipeline, refer to [MLPipeline](APIdocs/MLPipeline/merged-MLPipeline.md) for usage. 

---

## Train a Model from Scratch

The procedure of training a model from scratch usually involves following steps:

1. define your model (by connecting layers/activations into a network)
2. decide your loss function (which function to optimize)
3. optimization (choose a proper algorithm and hyper parameters, and train)
4. evaluation (evalute your model) 

Before diving into details, please make sure BigDL is installed, BigDL engine initialized properly, and your data is in proper format. Refer to [Before using BigDL](#before-using-bigdl) and [Prepare Your Data](#prepare-your-data) for details.  

The most recommended way to create your first model is to modify from an existing one. BigDL provides plenty of models for you to refer to. See [Scala Models/Examples](UserGuide/resources.md) and [Python Models/Examples and Tutorials](PythonSupport/python-resources.md). 

To define a model, you can either use the Sequential API or Functional API. The Functional API is more flexible than Sequential API. Refer to [Sequential API](APIdocs/Model/Sequential.md) and [Functional API](APIdocs/Model/Functional.md) for how to define models in different shapes.  

After creating the model, you will have to deside which loss function to use in training. Find the details of losses defined in BigDL in [Losses](APIdocs/Losses/merged-Losses.md).  

Now you create an `Optimizer` and set the loss function, input dataset along with other hyper parameters into the Optimizer. Then call `Optimizer.optimize` to train. Refer to [Optimizer docs](APIdocs/Optimizers/DistriOptimizer.md) for details. 

Evaluation can be performed periodically during a training. Before calling `Optimizer.optimize`, use `Optimizer.setValidation` (in Scala) or `Optimizer.set_validation` (in Python) to set validation configurations, e.g. validation dataset, validation metrics, etc. For a list of defined metrics, refer to [Metrics](APIdocs/Metrics/merged-Metrics.md).

When `Optimizer.optimize` finishes, it will return a trained model. You can then use `module.predict` for predictions. Refer to [Module API docs](APIdocs/Model/ModuleAPI.md) for detailed usage.    

## Stop and Resume a Training

Training a deep learning model sometimes takes a very long time. It may be stopped or interrupted and we need the training to resume from where we have left. 

To enable this, you have to configure `Optimizer` to periodically dump the optimized model and optimization state into snapshot files. Use `Optimizer.setCheckpoint` (in Scala) or `optimizer.set_checkpoint` (in Python) to configure the frequency and paths of writing snapshots. Then during your training, you will find several snapshot files written in the checkpoint path. Refer to [Optimizer API](APIdocs/Optimizers/DistriOptimizer.md) for details. 

Later, after the training stops, you can resume from any saved point. Choose one of the model snapshots and the corresponding optimization state to resume (the iteration number of the the snapshots is in the file name suffix). Use `Module.load` to load the model snapshot into an object, and `OptimMethod.load` to load optimization state into an object. Then create a new `Optimizer` with the loaded model and optim method. Call `Optimizer.optimize`, you will resume from the point where the snapshot is taken. Refer to [OptimMethod API]() and [Module API](APIdocs/Model/ModuleAPI.md) for details.
 
You can also resume training without loading the optimization state, if you intend to change the learning rate schedule or even optimization method. Just create an `Optimizer` with loaded model and a new instance of OptimMethod. 


--- 

## Use Pre-trained Models/Layers

Pre-train is a useful strategy in training deep learning models. You may use the pre-trained features (e.g. embeddings) in your model, or do a fine-tuning for a different dataset or target.
 
To use a learnt model as a whole, you can use `Module.load` to load the entire model, Then create an `Optimizer` with the loaded model set into it. Refer to [Optmizer API](APIdocs/Optimizers/DistriOptimizer.md) and [Module API](APIdocs/Model/ModuleAPI.md) for details. 

Instead of using an entire model, you can also use pre-trained weights/biases in certain layers. After a `Module`(Scala) or `Layer`(Python) is created, use `Module.setWeightsBias` (in Scala) or `layer.set_weights` (in Python) to initialize the weights with pre-trained weights. Then continue to train your model as usual. 


---

## Monitor your training

You can use several methods

 * Visualization

BigDL writes the statistics as protobuf format and can be directly visualized using tensorboard. For details, refer to [Visualize the Training](ProgrammingGuide/visualization-with-tensorboard.md). 

 * Logging

In the training, BigDL provide a straight forward logging like this. You can see epoch/iteration/loss/throughput directly from the log.

```
 2017-01-10 10:03:55 INFO  DistriOptimizer$:241 - [Epoch 1 0/5000][Iteration 1][Wall Clock XXX] Train 512 in   XXXseconds. Throughput is XXX records/second. Loss is XXX.
 2017-01-10 10:03:58 INFO  DistriOptimizer$:241 - [Epoch 1 512/5000][Iteration 2][Wall Clock XXX] Train 512    in XXXseconds. Throughput is XXX records/second. Loss is XXX.
 2017-01-10 10:04:00 INFO  DistriOptimizer$:241 - [Epoch 1 1024/5000][Iteration 3][Wall Clock XXX] Train 512   in XXXseconds. Throughput is XXX records/second. Loss is XXX.
```

The DistriOptimizer log level is INFO by default. We implement a method named with `redirectSparkInfoLogs`  in `spark/utils/LoggerFilter.scala`. You can import and redirect at first.

```scala
 import com.intel.analytics.bigdl.utils.LoggerFilter
 LoggerFilter.redirectSparkInfoLogs()
```

This method will redirect all logs of `org`, `akka`, `breeze` to `bigdl.log` with `INFO` level, except `org.  apache.spark.SparkContext`. And it will output all `ERROR` message in console too.

You can disable the redirection with java property `-Dbigdl.utils.LoggerFilter.disable=true`. By default,   it will do redirect of all examples and models in our code.

You can set where the `bigdl.log` will be generated with `-Dbigdl.utils.LoggerFilter.logFile=<path>`. By    default, it will be generated under current workspace.

---

## Tuning

There're several strategies that may be useful when tuning an optimization. 

 * Change the learning Rate Schedule in SGD. Refer to [SGD docs](APIdocs/Optimizers/Optim-Methods/merged-Optim-Methods.md#sgd) for details. 
 * If overfit is seen, try use Regularization. Refer to [Regularizers](APIdocs/Regularizers/merged-Regularizers.md). 
 * Try change the initialization methods. Refer to [Initailizers](APIdocs/Initializers/merged-Initializers.md).
 * Try Adam or Adagrad at the first place. If they can't achive a good score, use SGD and find a proper learning rate schedule - it usually takes time, though. RMSProp is recommended for RNN models. Refer to [Optimization Algorithms](APIdocs/Optimizers/Optim-Methods/merged-Optim-Methods.md) for a list of supported optimization methods. 
