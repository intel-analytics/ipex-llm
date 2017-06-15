---
# **Examples**
---

This page shows how to build simple deep learning programs using BigDL, including:

1. [Training LeNet on MNIST](#training-lenet-on-mnist) - the "hello world" for deep learning
2. [Text Classification](#text-classification---working-with-spark-rdd) - working with Spark RDD transformations
3. [Image Classification](#image-classification---working-with-spark-dataframe-and-ml-pipeline) - working with Spark DataFrame and ML pipeline
4. [Python Text Classifier](#tutorial-text-classification-using-bigdl-python-api) - text classification using BigDL Python APIs
5. [BigDL Tutorials Notebooks](https://github.com/intel-analytics/BigDL-Tutorials) - A series of notebooks that step-by-step introduce you how to do data science on Apache Spark and BigDL framework
6. [Jupyter Notebook Tutorial](https://github.com/intel-analytics/BigDL/blob/branch-0.1/pyspark/dl/example/tutorial/simple_text_classification/text_classfication.ipynb) - using BigDL Python APIs in Jupyter notebook

## **Training LeNet on MNIST**
This tutorial is an explanation of what is happening in the [lenet](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/lenet/Train.scala) example, which trains [LeNet-5](http://yann.lecun.com/exdb/lenet/) on the [MNIST data](http://yann.lecun.com/exdb/mnist/) using BigDL.

A BigDL program starts with `import com.intel.analytics.bigdl._`; it then _**creates the `SparkContext`**_ using the `SparkConf` returned by the `Engine`; after that, it _**initializes the `Engine`**_.
````scala
  val conf = Engine.createSparkConf()
      .setAppName("Train Lenet on MNIST")
      .set("spark.task.maxFailures", "1")
  val sc = new SparkContext(conf)
  Engine.init
````
````Engine.createSparkConf```` will return a ````SparkConf```` populated with some appropriate configuration. And ````Engine.init```` will verify and read some environment information(e.g. executor numbers and executor cores) from the ````SparkContext````. You can find more information about the initialization in the [Programming Guilde](https://github.com/intel-analytics/BigDL/wiki/Programming-Guide#engine)

After the initialization, we need to:

1. _**Create the LeNet model**_ by calling the [````LeNet5()````](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/lenet/LeNet5.scala), which creates the LeNet-5 convolutional network model as follows:

````scala
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
````
2. Load the data by _**creating the [```DataSet```](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/dataset)**_ (either a distributed or local one depending on whether it runs on Spark or not), and then _**applying a series of [```Transformer```](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/dataset)**_ (e.g., ````SampleToGreyImg````, ````GreyImgNormalizer```` and ````GreyImgToBatch````):

````scala
    val trainSet = (if (sc.isDefined) {
        DataSet.array(load(trainData, trainLabel), sc.get, param.nodeNumber)
      } else {
        DataSet.array(load(trainData, trainLabel))
      }) -> SampleToGreyImg(28, 28) -> GreyImgNormalizer(trainMean, trainStd) -> GreyImgToBatch(
        param.batchSize)
````

After that, we _**create the [```Optimizer```](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/optim)**_ (either a distributed or local one depending on whether it runs on Spark or not) by specifying the ````DataSet````, the model and the ````Criterion```` (which, given input and target, computes gradient per given loss function):
````scala
  val optimizer = Optimizer(
    model = model,
    dataset = trainSet,
    criterion = ClassNLLCriterion[Float]())
````

Finally (after optionally specifying the validation data and methods for the ````Optimizer````), we _**train the model by calling ````Optimizer.optimize()````**_:
````scala
  optimizer
    .setValidation(
      trigger = Trigger.everyEpoch,
      dataset = validationSet,
      vMethods = Array(new Top1Accuracy))
    .setState(state)
    .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
    .optimize()
````

## Text Classification - Working with Spark RDD
This tutorial describes the [text_classification](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/textclassification) example, which builds a text classifier using a simple convolutional neural network (CNN) model. (It was first described by [this Keras tutorial](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)).

After importing ```com.intel.analytics.bigdl._``` and some initialization, the [example](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/textclassification/TextClassifier.scala) broadcasts the pre-trained world embedding and loads the input data using RDD transformations:
````scala
  // For large dataset, you might want to get such RDD[(String, Float)] from HDFS
  val dataRdd = sc.parallelize(loadRawData(), param.partitionNum)
  val (word2Meta, word2Vec) = analyzeTexts(dataRdd)
  val word2MetaBC = sc.broadcast(word2Meta)
  val word2VecBC = sc.broadcast(word2Vec)
  val vectorizedRdd = dataRdd
      .map {case (text, label) => (toTokens(text, word2MetaBC.value), label)}
      .map {case (tokens, label) => (shaping(tokens, sequenceLen), label)}
      .map {case (tokens, label) => (vectorization(
        tokens, embeddingDim, word2VecBC.value), label)}
````

The [example](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/textclassification/TextClassifier.scala) then converts the processed data (`vectorizedRdd`) to an RDD of Sample, and randomly splits the sample RDD (`sampleRDD`) into training data (`trainingRDD`) and validation data (`valRDD`):
````scala
  val sampleRDD = vectorizedRdd.map {case (input: Array[Array[Float]], label: Float) =>
        Sample(
          featureTensor = Tensor(input.flatten, Array(sequenceLen, embeddingDim))
            .transpose(1, 2).contiguous(),
          labelTensor = Tensor(Array(label), Array(1)))
      }

  val Array(trainingRDD, valRDD) = sampleRDD.randomSplit(
    Array(trainingSplit, 1 - trainingSplit))
````

After that, the [example](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/textclassification/TextClassifier.scala) builds the CNN model, creates the ````Optimizer````, pass the RDD of training data (`trainingRDD`) to the ```Optimizer``` (with specific batch size), and finally trains the model (using ````Adagrad```` as the optimization method, and setting relevant hyper parameters in ````state````):
````scala
  val optimizer = Optimizer(
    model = buildModel(classNum),
    sampleRDD = trainingRDD,
    criterion = new ClassNLLCriterion[Float](),
    batchSize = param.batchSize
  )
  val state = T("learningRate" -> 0.01, "learningRateDecay" -> 0.0002)
  optimizer
    .setState(state)
    .setOptimMethod(new Adagrad())
    .setValidation(Trigger.everyEpoch, valRDD, Array(new Top1Accuracy[Float]), param.batchSize)
    .setEndWhen(Trigger.maxEpoch(2))
    .optimize()
````

## **Image Classification** - Working with Spark DataFrame and ML pipeline
This tutorial describes the [image_classification](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/imageclassification) example, which loads a BigDL ([Inception](http://arxiv.org/abs/1409.4842)) model or Torch ([Resnet](https://arxiv.org/abs/1512.03385)) model that is trained on [ImageNet](http://image-net.org/download-images) data, and then applies the loaded model to predict the contents of a set of images using BigDL and Spark [ML pipeline](https://spark.apache.org/docs/1.6.3/ml-guide.html).

After importing ```com.intel.analytics.bigdl._``` and some initialization, the [example](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/imageclassification/ImagePredictor.scala) first [loads](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/imageclassification/MlUtils.scala) the specified model:

````scala
  def loadModel[@specialized(Float, Double) T : ClassTag](param : PredictParams)
    (implicit ev: TensorNumeric[T]): Module[T] = {
    val model = param.modelType match {
      case TorchModel =>
        Module.loadTorch[T](param.modelPath)
      case BigDlModel =>
        Module.load[T](param.modelPath)
      case _ => throw new IllegalArgumentException(s"${param.modelType}")
    }
    model
  }
````

It then creates ```DLClassifer``` (a Spark ML pipelines [Transformer](https://spark.apache.org/docs/1.6.3/ml-pipeline.html#transformers)) that predicts the input value based on the specified deep learning model:
````scala
  val model = loadModel(param)
  val valTrans = new DLClassifier()
    .setInputCol("features")
    .setOutputCol("predict")

  val paramsTrans = ParamMap(
    valTrans.modelTrain -> model,
    valTrans.batchShape ->
    Array(param.batchSize, 3, imageSize, imageSize))
````

After that, the [example](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/imageclassification/ImagePredictor.scala)  loads the input images into a [DataFrame](https://spark.apache.org/docs/1.6.3/ml-guide.html#dataframe), and then predicts the class of each each image using the ```DLClassifer```:
````scala
  val valRDD = sc.parallelize(imageSet).repartition(partitionNum)
  val transf = RowToByteRecords() ->
      SampleToBGRImg() ->
      BGRImgCropper(imageSize, imageSize) ->
      BGRImgNormalizer(testMean, testStd) ->
      BGRImgToImageVector()

  val valDF = transformDF(sqlContext.createDataFrame(valRDD), transf)

  valTrans.transform(valDF, paramsTrans)
      .select("imageName", "predict")
      .show(param.showNum)
````
## **Tutorial: Text Classification using BigDL Python API**  

This tutorial describes the [textclassifier](https://github.com/intel-analytics/BigDL/tree/master/pyspark/dl/models/textclassifier) example written using BigDL Python API, which builds a text classifier using a CNN (convolutional neural network) or LSTM or GRU model (as specified by the user). (It was first described by [this Keras tutorial](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html))

The example first creates the `SparkContext` using the SparkConf` return by the `create_spark_conf()` method, and then initialize the engine:
```python
  sc = SparkContext(appName="text_classifier",
                    conf=create_spark_conf())
  init_engine()
```

It then loads the [20 Newsgroup dataset](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html) into RDD, and transforms the input data into an RDD of `Sample`. (Each `Sample` in essence contains a tuple of two NumPy ndarray representing the feature and label).

```python
  texts = news20.get_news20()
  data_rdd = sc.parallelize(texts, 2)
  ...
  sample_rdd = vector_rdd.map(
      lambda (vectors, label): to_sample(vectors, label, embedding_dim))
  train_rdd, val_rdd = sample_rdd.randomSplit(
      [training_split, 1-training_split])   
```

After that, the example creates the neural network model as follows:
```python
def build_model(class_num):
    model = Sequential()

    if model_type.lower() == "cnn":
        model.add(Reshape([embedding_dim, 1, sequence_len]))
        model.add(SpatialConvolution(embedding_dim, 128, 5, 1))
        model.add(ReLU())
        model.add(SpatialMaxPooling(5, 1, 5, 1))
        model.add(SpatialConvolution(128, 128, 5, 1))
        model.add(ReLU())
        model.add(SpatialMaxPooling(5, 1, 5, 1))
        model.add(Reshape([128]))
    elif model_type.lower() == "lstm":
        model.add(Recurrent()
                  .add(LSTM(embedding_dim, 128)))
        model.add(Select(2, -1))
    elif model_type.lower() == "gru":
        model.add(Recurrent()
                  .add(GRU(embedding_dim, 128)))
        model.add(Select(2, -1))
    else:
        raise ValueError('model can only be cnn, lstm, or gru')

    model.add(Linear(128, 100))
    model.add(Linear(100, class_num))
    model.add(LogSoftMax())
    return model
```
Finally the example creates the `Optimizer` (which accepts both the model and the training Sample RDD) and trains the model by calling `Optimizer.optimize()`:

```python
optimizer = Optimizer(
    model=build_model(news20.CLASS_NUM),
    training_rdd=train_rdd,
    criterion=ClassNLLCriterion(),
    end_trigger=MaxEpoch(max_epoch),
    batch_size=batch_size,
    optim_method="Adagrad",
    state=state)
...
train_model = optimizer.optimize()
```

---