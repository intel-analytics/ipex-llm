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

## **Dowload the Data**

We used [Pima Indians onset of diabetes](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) as dataset for the demo. It's a standard machine learning dataset from the UCI Machine Learning repository. It describes patient medical record data for Pima Indians and whether they had an onset of diabetes within five years.

For more details about the data, please refer [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names)

```
wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```
---

After the data is ready, we can now write the deep learning code with DLLib

## **Initialize DLLib NNContext**

We need do some initialization at first
```
import com.intel.analytics.bigdl.dllib.NNContext

val sc = NNContext.initNNContext("dllib_demo")
```

Then create Spark session so we can use Spark API to load and process the data
```
import org.apache.spark.sql.SQLContext
val spark = new SQLContext(sc)
```

## **Load and Process data using Spark API**

DLlib supports Spark Dataframes as the input to the distributed training, and as
the input/output of the distributed inference. Consequently, the user can easily
process large-scale dataset using Apache Spark, and directly apply AI models on
the distributed (and possibly in-memory) Dataframes without data conversion or serialization

#### Load the data into Spark DataFrame
```
val path = "pima-indians-diabetes.data.csv"
val df = spark.read.options(Map("inferSchema"->"true","delimiter"->",")).csv(path)
      .toDF("num_times_pregrant", "plasma_glucose", "blood_pressure", "skin_fold_thickness", "2-hour_insulin", "body_mass_index", "diabetes_pedigree_function", "age", "class")
```

#### Process Spark DataFrame

Process the DataFrame to create the label and split it into traing part and validation part

```
val df2 = df.withColumn("label",col("class").cast(DoubleType) + lit(1))
val Array(trainDF, valDF) = df2.randomSplit(Array(0.8, 0.2))
```

Now we have got the data which is ready to train. Next we will build a deep learning model using DLLib Keras API

## **Define Deep Learning Model**

The procedure of training a model from scratch usually involves following steps:

1. define your model (by connecting layers/activations into a network)
2. decide your loss function (which function to optimize)
3. train (choose a proper algorithm and hyper parameters, and train)
4. evaluation (evaluate your model)

To define a model, you can use the [Keras Style API](https://github.com/intel-analytics/BigDL/tree/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/keras). You may want to refer to [Lenet](https://github.com/intel-analytics/BigDL/blob/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/models/lenet/LeNet5.scala#L59) for how to define models.
```
import com.intel.analytics.bigdl.dllib.keras.layers._
val x1 = Input[Float](Shape(8))
val dense1 = Dense[Float](12, activation="relu").inputs(x1)
val dense2 = Dense[Float](8, activation="relu").inputs(dense1)
val dense3 = Dense[Float](2).inputs(dense2)
val dmodel = Model(x1, dense3)
```

After creating the model, you will have to decide which loss function to use in training. For a list of loss functions, refer to [loss function](https://github.com/intel-analytics/BigDL/tree/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/keras/objectives)

Now you can use `compile` function of the model to set the loss function, optimization method.
```
dmodel.compile(optimizer = new Adam[Float](),
      loss = ClassNLLCriterion[Float]())
```

Now the model is built and ready to train.

## **Train Deep Learning Model**
Now you can use 'fit' begin the training, please set the feature columns and label columns. Model Evaluation can be performed periodically during a training.
```
dmodel.fit(x=trainDF, batchSize=4, nbEpoch = 2,
  featureCols = Array("num_times_pregrant", "plasma_glucose", "blood_pressure",
    "skin_fold_thickness", "2-hour_insulin", "body_mass_index",
    "diabetes_pedigree_function", "age"), labelCols = Array("label"), valX = valDF
)
```

## **Inference**
After `fit` finishes, you can then use the trained model for prediction or evaluation.
```
dmodel.predict(df, featureCols = Array("num_times_pregrant", "plasma_glucose", "blood_pressure",
  "skin_fold_thickness", "2-hour_insulin", "body_mass_index",
  "diabetes_pedigree_function", "age"), predictionCol = "predict")
```

## **Save a Model**

When training is finished, you may need to save the final model for later use. 

BigDL allows you to save your BigDL model on local filesystem, HDFS, or Amazon s3.
```
val modelPath = "/tmp/keras.model"
dmodel.saveModel(modelPath)
```
---

## **Use Pre-trained Models/Layers**

Pre-train is a useful strategy when training deep learning models. You may use the pre-trained features (e.g. embeddings) in your model, or do a fine-tuning for a different dataset or target.

To use a learnt model as a whole, you can use `Models.loadModel` to load the entire model.

```
val loadModel = Models.loadModel[Float](modelPath)
```

## **Monitor your training**

BigDL provides a convenient way to monitor/visualize your training progress. It writes the statistics collected during training/validation and they can be visualized in real-time using tensorboard. These statistics can also be retrieved into readable data structures later and visualized in other tools (e.g. Jupyter notebook).
```
dmodel.setTensorBoard("./", "dllib_demo")
dmodel.fit(x=trainDF, batchSize=4, nbEpoch = 2,
  featureCols = Array("num_times_pregrant", "plasma_glucose", "blood_pressure",
    "skin_fold_thickness", "2-hour_insulin", "body_mass_index",
    "diabetes_pedigree_function", "age"), labelCols = Array("label"), valX = valDF
)
val rawTrain = dmodel.getTrainSummary("Loss")
val rawVal = dmodel.getValidationSummary("Loss")
```
---
