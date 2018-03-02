package com.intel.analytics.bigdl.example.DLframes

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dlframes.{DLClassifier, DLModel}
import com.intel.analytics.bigdl.example.DLframes.Utils.LocalParams
import com.intel.analytics.bigdl.example.imageclassification.MlUtils._
import com.intel.analytics.bigdl.example.imageclassification.RowToByteRecords
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.functions.{col, rand, udf}

object TransferLearning {

  def loadImages(path: String, partitionNum: Int, sqlContext: SQLContext): DataFrame = {

    val paths = LocalImageFiles.readPaths(Paths.get(path), hasLabel = false)
    val datardd = sqlContext.sparkContext.parallelize(imagesLoad(paths, 256), partitionNum)

    val transf = RowToByteRecords() ->
      BytesToBGRImg() ->
      BGRImgCropper(imageSize, imageSize) ->
      BGRImgNormalizer(testMean, testStd) ->
      BGRImgToImageVector()

    val sizeOfFeature = udf((features: org.apache.spark.mllib.linalg.Vector) => features.size)
    val createLabel = udf((name: String) => if (name.contains("cat")) 1.0 else 2.0)

    val imagesDF: DataFrame = transformDF(sqlContext.createDataFrame(datardd), transf)
      .withColumn("label", createLabel(col("imageName")))

    imagesDF
  }

  def getModel(params: LocalParams) = {

    val loaded: AbstractModule[Activity, Activity, Float] = Module.loadCaffeModel[Float](params.caffeDefPath, params.modelPath)
    val model = Sequential[Float]()
    model.add(loaded)
    model.add(Linear[Float](1000, 2)).add(ReLU()).add(LogSoftMax())
    model
  }

  def main(args: Array[String]): Unit = {

    val defaultParams = LocalParams(folder = "/Users/guoqiong/intelWork/projects/dlFrames/data/kaggle/train_100")

    Utils.parser.parse(args, defaultParams).map { params =>

      val conf = Engine.createSparkConf().setAppName("TransferLearning")
        .setMaster("local[2]")
      val spark = SparkSession.builder().config(conf).getOrCreate()
      Engine.init

      Logger.getLogger("org").setLevel(Level.ERROR)
      spark.sparkContext.setLogLevel("ERROR")

      val imagesDF: DataFrame = loadImages(params.folder, params.batchSize, spark.sqlContext)

      val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.2, 0.8), seed = 1)

      trainingDF.cache()
      validationDF.cache()

      val criterion = ClassNLLCriterion[Float]()

      val loaded: AbstractModule[Activity, Activity, Float] = Module.loadCaffeModel[Float](params.caffeDefPath, params.modelPath)
      val model = Sequential[Float]()
        .add(loaded)
        .add(Linear[Float](1000, 2)).add(ReLU()).add(LogSoftMax())

      val dlc: DLClassifier[Float] = new DLClassifier[Float](model, criterion, Array(3, 224, 224))
        .setBatchSize(4)
        .setOptimMethod(new Adam())
        .setLearningRate(1e-2)
        .setLearningRateDecay(1e-5)
        .setMaxEpoch(10)
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")

      val time1 = System.nanoTime()

      val dlModel: DLModel[Float] = dlc.fit(trainingDF)

      val time2 = System.nanoTime()

      val predictions = dlModel.setBatchSize(2).transform(validationDF)

      predictions.show(5)

      val toZero = udf { d: Double => if (d > 1) 1.0 else 0.0 }
      Evaluation.evaluate(predictions.withColumn("label", toZero(col("label")))
        .withColumn("prediction", toZero(col("prediction"))))

    }
  }

}


object Evaluation {

  def evaluate(evaluateDF: DataFrame) = {
    val binaryEva = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")
    val out1 = binaryEva.evaluate(evaluateDF)
    println("AUROC: " + out1)

    val multiEva = new MulticlassClassificationEvaluator().setMetricName("weightedPrecision")
    val out2 = multiEva.evaluate(evaluateDF)
    println("precision: " + out2)

    val multiEva2 = new MulticlassClassificationEvaluator().setMetricName("weightedRecall")
    val out3 = multiEva2.evaluate(evaluateDF)
    println("recall: " + out3)

    Seq(out1, out2, out3).map(x => x)

  }
}

