---
## **Setup Deep Learning Project using BigDL DLLib**

To use BigDL DLLib to build your own deep learning application, you can use maven to create your project and add bigdl-dllib to your dependency. Please add below code to your pom.xml to add BigDL DLLib as your dependency:
```
<dependency>
    <groupId>com.intel.analytics.bigdl</groupId>
    <artifactId>bigdl-dllib-spark_2.4.6</artifactId>
    <version>0.14.0-SNAPSHOT</version>
</dependency>
```

For more information about how to use BigDL to build your applications, please refer https://github.com/intel-analytics/BigDL/tree/branch-2.0/apps/SimpleMlp
---

## **Prepare your Data**

Your data can be Spark DataFrame for DLLib Keras Model training, evaluation and prediction. You can use Spark API to create Spark DataFrame and feed it into BigDL Keras Model.

---

## **Train a Model from Scratch**

The procedure of training a model from scratch usually involves following steps:

1. define your model (by connecting layers/activations into a network)
2. decide your loss function (which function to optimize)
3. train (choose a proper algorithm and hyper parameters, and train)
4. evaluation (evaluate your model) 

Before training models, please make sure BigDL DLLib is installed, BigDL context is initialized properly, and your data is ready. Refer to [Before using BigDL](#before-using-bigdl) and [Prepare Your Data](#prepare-your-data) for details.

To define a model, you can use the [Keras Style API](https://github.com/intel-analytics/BigDL/tree/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/keras). You may want to refer to [Lenet](https://github.com/intel-analytics/BigDL/blob/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/models/lenet/LeNet5.scala#L59) for how to define models.
```
import com.intel.analytics.bigdl.dllib.keras.models.Sequential
val model = Sequential[Float]()
model.add(Dense[Float](2, activation = "sigmoid", inputShape = Shape(6)))
```

After creating the model, you will have to decide which loss function to use in training. For a list of loss functions, refer to [loss function](https://github.com/intel-analytics/BigDL/tree/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/keras/objectives)

Now you can use `compile` function of the model to set the loss function, optimization method.
```
model.compile(optimizer = new SGD[Float](), loss = ClassNLLCriterion[Float]())
```

Next you can use 'fit' begin the training, please set the feature columns and label columns. Model Evaluation can be performed periodically during a training.
```
model.fit(df, batchSize = 4, nbEpoch = 1, featureCols = Array("f1", "f2", "f3"),
  labelCols = Array("label"))
```

After `fit` finishes, you can then use the trained model for prediction or evaluation.
```
model.predict(df, featureCols = Array("f1", "f2", "f3"), predictionCol = "predict")
```

---

## **Save a Model**

When training is finished, you may need to save the final model for later use. 

BigDL allows you to save your BigDL model on local filesystem, HDFS, or Amazon s3 (refer to [Model Save](APIGuide/Module.md#model-save)). 
```
model.saveModule(path_to_save)
```
---

## **Use Pre-trained Models/Layers**

Pre-train is a useful strategy when training deep learning models. You may use the pre-trained features (e.g. embeddings) in your model, or do a fine-tuning for a different dataset or target.

To use a learnt model as a whole, you can use `Module.loadModule` to load the entire model.

```
val model = Module.loadModule[Float](path_to_save)
```

## **Monitor your training**


BigDL provides a convenient way to monitor/visualize your training progress. It writes the statistics collected during training/validation and they can be visualized in real-time using tensorboard. These statistics can also be retrieved into readable data structures later and visualized in other tools (e.g. Jupyter notebook).
```
model.setTensorBoard("./", "testTensorBoard")
model.fit(trainingData, batchSize = 8, nbEpoch = 2, validationData = trainingData)

val rawTrain = model.getTrainSummary("Loss")
val rawVal = model.getValidationSummary("Loss")
```
---
