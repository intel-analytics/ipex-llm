package com.intel.analytics.bigdl.example.DLframes.imageInference

import com.intel.analytics.bigdl.example.DLframes.imageInference.Utils.LocalParams
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

object LogisticRegressionExp {

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

      val createLabel = udf((name: String) => if (name.contains("cat")) 1.0 else 0.0)
      val imagesDF1: DataFrame = Utils.loadImages3(params.folder, params.batchSize,  spark.sqlContext)
        .withColumn("label", createLabel(col("imageName")))
     //   .withColumnRenamed("features", "imageFeatures")
      //.sample(true, 0.01)

      imagesDF1.show(10)
      imagesDF1.printSchema()
//      val imagesDF = org.apache.spark.mllib.util.MLUtils.convertVectorColumnsToML(imagesDF1, "imageFeatures")
//        .withColumnRenamed("imageFeatures", "features")


      val Array(validationDF, trainingDF) = imagesDF1.randomSplit(Array(0.90, 0.10), seed = 1L)

      trainingDF.groupBy("label").count().show()
      validationDF.groupBy("label").count().show()
      trainingDF.persist()
      validationDF.persist()

      // Load training data
      val training = spark.read.format("libsvm").load("/Users/guoqiong/intelWork/git/spark/data/mllib/sample_libsvm_data.txt")


      training.printSchema()
      training.show(5)

      training.rdd.take(1).map(row => println(row(1).getClass))

      val lr = new LogisticRegression()
        .setMaxIter(10)
        .setRegParam(0.3)
        .setElasticNetParam(0.8)

      // Fit the model
      val lrModel = lr.fit(validationDF)
      val predictions = lrModel.transform(validationDF)
      predictions.printSchema()
      predictions.select("imageName", "rawPrediction", "probability", "prediction")
        .show(100, false)
    }
  }
}