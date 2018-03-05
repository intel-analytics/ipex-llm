package com.intel.analytics.bigdl.example.DLframes

import breeze.numerics.exp
import com.intel.analytics.bigdl.dlframes.{DLClassifier, DLModel}
import com.intel.analytics.bigdl.example.DLframes.Utils.LocalParams
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, udf}

object TransferLearning {

  def main(args: Array[String]): Unit = {

    //TODO delete
    val defaultParams = LocalParams(folder = "/Users/guoqiong/intelWork/projects/dlFrames/data/kaggle/train_100")

    Utils.parser.parse(args, defaultParams).map { params =>

      val conf = Engine.createSparkConf().setAppName("TransferLearning")
        .setMaster("local[8]")
      val spark = SparkSession.builder().config(conf).getOrCreate()
      Engine.init

      Logger.getLogger("org").setLevel(Level.ERROR)
      spark.sparkContext.setLogLevel("ERROR")

      val createLabel = udf((name: String) => if (name.contains("cat")) 1.0 else 2.0)
      val imagesDF: DataFrame = Utils.loadImages(params.folder, params.batchSize, spark.sqlContext)
        .withColumn("label", createLabel(col("imageName")))
      //.sample(true, 0.01)

      val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.2, 0.8), seed = 1L)

      trainingDF.groupBy("label").count().show()
      validationDF.groupBy("label").count().show()
      trainingDF.persist()
      validationDF.persist()

      val criterion = ClassNLLCriterion[Float]()

      val loaded: AbstractModule[Activity, Activity, Float] = Module
        .loadCaffeModel[Float](params.caffeDefPath, params.modelPath)
      val model = Sequential[Float]().add(loaded)
        .add(Linear[Float](1000, 2)).add(ReLU()).add(LogSoftMax())

      val dlc: DLClassifier[Float] = new DLClassifier[Float](model, criterion, Array(3, 224, 224))
        .setBatchSize(params.batchSize)
        .setOptimMethod(new Adam())
        .setLearningRate(1e-2)
        .setLearningRateDecay(1e-5)
        .setMaxEpoch(params.nEpochs)
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")

      val time1 = System.nanoTime()

      val dlModel: DLModel[Float] = dlc.fit(trainingDF)

      val time2 = System.nanoTime()

      val count = validationDF.count().toInt
      println("training time: " + (time2 - time1) * (1e-9) + "s")
      validationDF.show(100)
      val predictions = dlModel.setBatchSize(params.batchSize).transform(validationDF.limit(count))

      predictions.persist()
      predictions.show(100)

      val evaluation = new MulticlassClassificationEvaluator().setPredictionCol("prediction")
        .setMetricName("accuracy").evaluate(predictions)
      println("evaluation result on validationDF: " + evaluation)
    }
  }

}