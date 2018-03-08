package com.intel.analytics.bigdl.example.DLframes.imageInference

import com.intel.analytics.bigdl.dlframes.{DLClassifier, DLModel}
import com.intel.analytics.bigdl.example.DLframes.imageInference.Utils.LocalParams
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

object TransferLearning {

  def main(args: Array[String]): Unit = {

    //TODO delete
    val defaultParams = LocalParams(folder = "/Users/guoqiong/intelWork/projects/dlFrames/data/kaggle/train_100",
      caffeDefPath = "/Users/guoqiong/intelWork/git/caffe/models/bvlc_googlenet/deploy.prototxt")

    Utils.parser.parse(args, defaultParams).map { params =>

      val conf = Engine.createSparkConf().setAppName("TransferLearning")
        .setMaster("local[8]")
      val spark = SparkSession.builder().config(conf).getOrCreate()
      Engine.init

      Logger.getLogger("org").setLevel(Level.ERROR)
      spark.sparkContext.setLogLevel("ERROR")

      val createLabel = udf((name: String) => if (name.contains("cat")) 1.0 else 2.0)
      val imagesDF: DataFrame = Utils.loadImages3(params.folder, params.batchSize, spark.sqlContext)
        .withColumn("label", createLabel(col("imageName")))
      //.sample(true, 0.01)

      val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.20, 0.80), seed = 1L)

      trainingDF.groupBy("label").count().show()
      validationDF.groupBy("label").count().show()
      trainingDF.persist()
      validationDF.persist()


      //val criterion = CrossEntropyCriterion[Float]()
      //val criterion = MSECriterion[Float]()
      val criterion = ClassNLLCriterion[Float]()

      val loaded: AbstractModule[Activity, Activity, Float] = Module
        .loadCaffeModel[Float](params.caffeDefPath, params.modelPath)
      val model = Sequential[Float]().add(loaded)
        .add(Linear[Float](1000, 2)).add(ReLU()).add(LogSoftMax())

      val dlc: DLClassifier[Float] = new DLClassifier[Float](model, criterion, Array(3, 224, 224))
        .setBatchSize(params.batchSize)
        .setOptimMethod(new Adam())
        .setLearningRate(1e-2)
        .setLearningRateDecay(1e-6)
        .setMaxEpoch(params.nEpochs)
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")

      val time1 = System.nanoTime()

      val dlModel: DLModel[Float] = dlc.fit(validationDF)

      val time2 = System.nanoTime()

      val count1 = trainingDF.count().toInt
      val count2 = validationDF.count().toInt

      println("training time: " + (time2 - time1) * (1e-9) + "s")

      val predictions = dlModel.setBatchSize(params.batchSize).transform(validationDF.limit(count1))
      predictions.show(200)

      val evaluation = new MulticlassClassificationEvaluator().setPredictionCol("prediction")
        .setMetricName("accuracy").evaluate(predictions)
      println("evaluation result on validationDF: " + evaluation)
    }
  }

}