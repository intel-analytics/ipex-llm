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

val sc = NNContext.initNNContext("dllib demo")
```

Then create Spark session so we can use Spark API to load and process the data
```
import org.apache.spark.sql.{DataFrame, SQLContext}
val sqlContext = new SQLContext(sc)
```

## **Load and Process data using Spark API**

DLlib supports Spark Dataframes as the input to the distributed training, and as
the input/output of the distributed inference. Consequently, the user can easily
process large-scale dataset using Apache Spark, and directly apply AI models on
the distributed (and possibly in-memory) Dataframes without data conversion or serialization

#### Load the data into Spark DataFrame
```
val path = "pima-indians-diabetes.data.csv"
df = spark.read.csv(path, sep=',', inferSchema=True).toDF("num_times_pregrant", "plasma_glucose", "blood_pressure", "skin_fold_thickness", "2-hour_insulin", "body_mass_index", "diabetes_pedigree_function", "age", "class")
```

#### Process Spark DataFrame
```
vecAssembler = VectorAssembler(outputCol="features")
vecAssembler.setInputCols(["num_times_pregrant", "plasma_glucose", "blood_pressure", "skin_fold_thickness", "2-hour_insulin", "body_mass_index", "diabetes_pedigree_function", "age"])
train_df = vecAssembler.transform(df)

changedTypedf = train_df.withColumn("label", train_df["class"].cast(DoubleType())+lit(1))\
    .select("features", "label")
(trainingDF, validationDF) = changedTypedf.randomSplit([0.9, 0.1])
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
import com.intel.analytics.bigdl.dllib.keras.models.Model
x1 = Input(shape=(8,))
dense1 = Dense(12, activation='relu')(x1)
dense2 = Dense(8, activation='relu')(dense1)
dense3 = Dense(2)(dense2)
model = Model(x1, dense3)
```

After creating the model, you will have to decide which loss function to use in training. For a list of loss functions, refer to [loss function](https://github.com/intel-analytics/BigDL/tree/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/keras/objectives)

Now you can use `compile` function of the model to set the loss function, optimization method.
```
model.compile(optimizer = new Adam[Float](), loss = ClassNLLCriterion[Float]())
```

Now the model is built and ready to train.

## **Train Deep Learning Model**
Now you can use 'fit' begin the training, please set the feature columns and label columns. Model Evaluation can be performed periodically during a training.
```
model.fit(df, batchSize = 4, nbEpoch = 1, featureCols = Array("f1", "f2", "f3"),
  labelCols = Array("label"))
```

## **Inference**
After `fit` finishes, you can then use the trained model for prediction or evaluation.
```
model.predict(df, featureCols = Array("f1", "f2", "f3"), predictionCol = "predict")
```

## **Save a Model**

When training is finished, you may need to save the final model for later use. 

BigDL allows you to save your BigDL model on local filesystem, HDFS, or Amazon s3 (refer to [Model Save](APIGuide/Module.md#model-save)). 
```
val path = "/tmp/keras.model"
model.saveModule(path)
```
---

## **Use Pre-trained Models/Layers**

Pre-train is a useful strategy when training deep learning models. You may use the pre-trained features (e.g. embeddings) in your model, or do a fine-tuning for a different dataset or target.

To use a learnt model as a whole, you can use `Module.loadModule` to load the entire model.

```
val model = Module.loadModule[Float](path)
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
