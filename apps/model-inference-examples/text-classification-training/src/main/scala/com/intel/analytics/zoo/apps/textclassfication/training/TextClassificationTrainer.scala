package com.intel.analytics.zoo.apps.textclassfication.training

import java.io.File

import ch.qos.logback.classic.{Level, Logger => LogbackLogger}
import com.intel.analytics.bigdl.dataset.{Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.optim.{Adagrad, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.apps.textclassfication.processing.TextProcessing
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.metrics.Accuracy
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import com.intel.analytics.zoo.pipeline.estimator.Estimator
import org.apache.spark.SparkConf
import org.slf4j.{Logger, LoggerFactory}
import scopt.OptionParser

import scala.collection.mutable.{Map => MutableMap}
import scala.io.Source

case class TextClassificationTrainerParams(trainDataDir: String = "./20news-18828",
                                           embeddingFile: String = "./glove/glove.6B.300d.txt",
                                           partitionNum: Int = 8,
                                           batchSize: Int = 256,
                                           nbEpoch: Int = 20,
                                           modelSaveDirPath: String = "./"
                                          )

object TextClassificationTrainer extends TextProcessing {

  val log: Logger = LoggerFactory.getLogger(this.getClass)
  val root: LogbackLogger = LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME).asInstanceOf[LogbackLogger]
  root.setLevel(Level.INFO)
  val sparkContextLoger = LoggerFactory.getLogger("org.apache.spark").asInstanceOf[LogbackLogger]
  sparkContextLoger.setLevel(Level.WARN)

  def loadTrainData(dirFile: File): (List[(String, Float)], Int) = {
    val categoryPathList = dirFile.listFiles().filter(_.isDirectory).toList.sorted
    val data = categoryPathList.map(categoryPath => {
      val labelId = categoryPathList.indexOf(categoryPath) + 0.0f
      categoryPath.listFiles().sorted.map(file => {
        val source = Source.fromFile(file, "ISO-8859-1")
        val text = try source.getLines().mkString("\n") finally source.close()
        (text, labelId)
      })
    }).flatten
    val classNum = categoryPathList.length
    log.info(s"Found ${data.length} texts.")
    log.info(s"Found ${classNum} classes")
    (data, classNum)
  }

  def buildModel(embeddingFile: File, wordToIndexMap: MutableMap[String, Int], sequenceLength: Int, classNum: Int, encoderOutputDim: Int = 256)(implicit ev: TensorNumeric[Float]) = {
    val model = Sequential[Float]()
    val embedding = WordEmbedding(embeddingFile.getAbsolutePath, wordToIndexMap.toMap, inputLength = sequenceLength)
    model.add(embedding)
    model.add(Convolution1D(encoderOutputDim, 5, activation = "relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.add(Dense(classNum, activation = "softmax"))
    model
  }

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[TextClassificationTrainerParams]("Text Classification Example") {
      opt[String]("trainDataDir")
        .required()
        .text("the base directory contains the training data")
        .action((x, params) => params.copy(trainDataDir = x))
      opt[String]("embeddingFile")
        .required()
        .text("the glove file")
        .action((x, params) => params.copy(embeddingFile = x))
      opt[Int]("partitionNum")
        .text("the number of partitions to cut the dataset into")
        .action((x, c) => c.copy(partitionNum = x))
      opt[Int]('b', "batchSize")
        .text("the number of samples per gradient update")
        .action((x, c) => c.copy(batchSize = x))
      opt[Int]("nbEpoch")
        .text("the number of iterations to train the model")
        .action((x, c) => c.copy(nbEpoch = x))
      opt[String]("modelSaveDirPath")
        .required()
        .text("the base directory to save model")
        .action((x, params) => params.copy(modelSaveDirPath = x))
    }

    parser.parse(args, TextClassificationTrainerParams()).map(params => {
      println(params)
      val trainDataDirFile = new File(params.trainDataDir)
      require(trainDataDirFile.exists(), "train data dir dose not exist")
      val embeddingFile = new File(params.embeddingFile)
      require(embeddingFile.exists(), "glove data file dose not exist")
      val stopWordsCount = 10
      val sequenceLength = 200
      val partitionNum = params.partitionNum
      val batchSize = params.batchSize
      val nbEpoch = params.nbEpoch
      val modelSaveDirPath = params.modelSaveDirPath

      val wordToIndexMap = doLoadWordToIndexMap(embeddingFile)
      log.info("wordToIndexMap size " + wordToIndexMap.size)

      val conf = new SparkConf()
        .setAppName("Text Classification Example")
        .setMaster("local[*]")
        .set("spark.task.maxFailures", "1")
      val sc = NNContext.initNNContext(conf)
      val wordToIndexMapBroadcast = sc.broadcast(wordToIndexMap)

      val (data, classNum) = loadTrainData(trainDataDirFile)
      val texts = sc.parallelize(data, params.partitionNum)
      val tensors = texts.map { case (text, label) => (doPreprocess(text, stopWordsCount, sequenceLength, wordToIndexMapBroadcast.value), label) }
      val samples = tensors.map { case (tensor: Tensor[Float], label: Float) => Sample(featureTensor = tensor, label = label) }
      val Array(trainings, validations) = samples.randomSplit(Array(0.8, 1 - 0.8))

      val model = buildModel(embeddingFile, wordToIndexMap, sequenceLength, classNum)

      val optimMethod = new Adagrad[Float](learningRate = 0.01, learningRateDecay = 0.001)
      val (checkpointTrigger, endTrigger) = (Trigger.everyEpoch, Trigger.maxEpoch(nbEpoch))
      val sample2batch = SampleToMiniBatch[Float](batchSize)
      val trainSet = FeatureSet.rdd(trainings.cache()) -> sample2batch
      val valSet = FeatureSet.rdd(validations.cache()) -> sample2batch

      val estimator = Estimator[Float](model, optimMethod)

      estimator.train(trainSet, SparseCategoricalCrossEntropy[Float](), Some(endTrigger),
        Some(checkpointTrigger), valSet, Array(new Accuracy[Float]()))

      val results = model.predict(validations)
      validations.take(5).foreach(x => println(x.label()))
      results.take(5)
      val resultClasses = model.predictClasses(validations)
      println("First five class predictions (label starts from 0):")
      resultClasses.take(5).foreach(println)

      model.saveModule(modelSaveDirPath, overWrite = true)

      sc.stop()
    })
  }

}
