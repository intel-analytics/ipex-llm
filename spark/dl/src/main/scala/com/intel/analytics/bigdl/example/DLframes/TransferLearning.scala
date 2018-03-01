package com.intel.analytics.bigdl.example.DLframes

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dlframes.{DLClassifier, DLClassifierModel, DLModel}
import com.intel.analytics.bigdl.example.imageclassification.MlUtils._
import com.intel.analytics.bigdl.example.imageclassification.RowToByteRecords
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.LabeledPoint
import scopt.OptionParser
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.functions.{col, rand, udf}


case class TransferParam(caffeDefPath: String = "/Users/guoqiong/intelWork/git/caffe/models/bvlc_googlenet/deploy.prototxt",
                         modelPath: String = "/Users/guoqiong/intelWork/projects/dlFrames/model/caffe/bvlc_googlenet.caffemodel",
                         folder: String = "/Users/guoqiong/intelWork/projects/dlFrames/data/kaggle/train_100/",
                         batchSize: Int = 8
                        )

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

    val datadf: DataFrame = transformDF(sqlContext.createDataFrame(datardd), transf)
      .withColumn("featureSize", sizeOfFeature(col("features")))
      .withColumn("label", createLabel(col("imageName")))

    val convertDF = org.apache.spark.mllib.util.MLUtils.convertVectorColumnsToML(datadf, "features")

    df2LP2(convertDF)
  }

  val df2LP2: (DataFrame) => DataFrame = df => {
    import df.sparkSession.implicits._
    df.select("features", "label").rdd.map { r =>

      LabeledPoint(r.getDouble(1), r.get(0).asInstanceOf[org.apache.spark.ml.linalg.Vector])
    }.toDF().orderBy(rand()).cache()
  }

  def getModel(params: TransferParam) = {

    val loaded: AbstractModule[Activity, Activity, Float] = Module.loadCaffeModel[Float](params.caffeDefPath, params.modelPath)
    val model = Sequential[Float]()
    model.add(loaded)
    model.add(Linear[Float](1000, 2)).add(ReLU()).add(LogSoftMax())
    model
  }

  def main(args: Array[String]): Unit = {

    val defaultParams = TransferParam()

    val parser = new OptionParser[TransferParam]("BigDL Example") {
      opt[String]("caffeDefPath")
        .text(s"caffeDefPath")
        .action((x, c) => c.copy(caffeDefPath = x))
      opt[String]("modelPath")
        .text(s"modelPath")
        .action((x, c) => c.copy(modelPath = x))
      opt[String]("folder")
        .text(s"folder")
        .action((x, c) => c.copy(folder = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: TransferParam) = {

    val conf = Engine.createSparkConf().setAppName("ModelInference")
      .setMaster("local[8]")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    Engine.init

    Logger.getLogger("org").setLevel(Level.ERROR)
    spark.sparkContext.setLogLevel("ERROR")

    val imagesDF: DataFrame = loadImages(params.folder, params.batchSize, spark.sqlContext)

    println("_______________________")
    imagesDF.printSchema()
    imagesDF.show()

    val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.2, 0.8), seed = 1)
    val model = getModel(params)

    println("__________________count")
    println(validationDF.count())

    println(trainingDF.count())
   // val criterion = ClassNLLCriterion[Float]()
    val criterion = MSECriterion[Float]()

    //val dlc: DLClassifier[Float] = new DLClassifier[Float](model, criterion, Array(150528))
    val dlc: DLClassifier[Float] = new DLClassifier[Float](model, criterion, Array(3, 224, 224))
      .setBatchSize(8)
      .setOptimMethod(new Adam())
      .setLearningRate(1e-2)
      .setLearningRateDecay(1e-5)
      .setMaxEpoch(10)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("predict")

    val dlModel: DLModel[Float] = dlc.fit(trainingDF)

    println("featuresize " + dlModel.featureSize)
    println("model weights  " + dlModel.model.getParameters())
    val time2 = System.nanoTime()

    val predictions = dlModel.setBatchSize(8).transform(validationDF)
    predictions.show(5)
  }

}
