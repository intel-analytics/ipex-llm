---

This section is a short introduction of some classic examples/tutorials. They can give you a clear idea of how to build simple deep learning programs using BigDL. Besides these examples, BigDL also provides plenty of models ready for re-use and examples in both Scala and Python - refer to [Resources](resources.md) section for details. 

---
## **Training LeNet on MNIST - The "hello world" for deep learning**
This tutorial is an explanation of what is happening in the [lenet](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/lenet/Train.scala) example, which trains [LeNet-5](http://yann.lecun.com/exdb/lenet/) on the [MNIST data](http://yann.lecun.com/exdb/mnist/) using BigDL. **Please open [Train.scala](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/lenet/Train.scala) to follow the example and check [README.md](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/lenet) for details of how to run the example**

A BigDL program starts with `import com.intel.analytics.bigdl._`; it then _**creates the `SparkContext`**_ using the `SparkConf` returned by the `Engine`; after that, it _**initializes the `Engine`**_.
````scala
  val conf = Engine.createSparkConf()
      .setAppName("Train Lenet on MNIST")
      .set("spark.task.maxFailures", "1")
  val sc = new SparkContext(conf)
  Engine.init
````
````Engine.createSparkConf```` will return a ````SparkConf```` populated with some appropriate configuration. And ````Engine.init```` will verify and read some environment information(e.g. executor numbers and executor cores) from the ````SparkContext````. 

After the initialization, we need to:

1._**Create the LeNet model**_ by calling the [````LeNet5()````](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/lenet/LeNet5.scala), which creates the LeNet-5 convolutional network model as follows:

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
2.Load the data by _**creating the [```DataSet```](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/dataset)**_ (either a distributed or local one depending on whether it runs on Spark or not), and then _**applying a series of [```Transformer```](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/dataset)**_ (e.g., ````SampleToGreyImg````, ````GreyImgNormalizer```` and ````GreyImgToBatch````):

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
    .setOptimMethod(new Adagrad(learningRate=0.01, learningRateDecay=0.0002))
    .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
    .optimize()
````

---
## **Text Classification - Working with Spark RDD**

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
  optimizer
    .setOptimMethod(new Adagrad(learningRate=0.01, learningRateDecay=0.0002))
    .setValidation(Trigger.everyEpoch, valRDD, Array(new Top1Accuracy[Float]), param.batchSize)
    .setEndWhen(Trigger.maxEpoch(2))
    .optimize()
````

---
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

It then creates ```DLClassifer``` (a Spark ML pipelines [Transformer](https://spark.apache.org/docs/latest/ml-pipeline.html#transformers)) that predicts the input value based on the specified deep learning model:
````scala
  val model = loadModel(param)
  val valTrans = new DLClassifierModel(model, Array(3, imageSize, imageSize))
    .setBatchSize(param.batchSize)
    .setFeaturesCol("features")
    .setPredictionCol("predict")
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

