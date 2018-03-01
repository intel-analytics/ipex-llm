package com.intel.analytics.bigdl.example.DLframes

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dlframes.{DLClassifierModel, DLModel}
import com.intel.analytics.bigdl.example.imageclassification.MlUtils._
import com.intel.analytics.bigdl.example.imageclassification.RowToByteRecords
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import scopt.OptionParser


case class InferenceParam(caffeDefPath: String = "/Users/guoqiong/intelWork/git/caffe/models/bvlc_googlenet/deploy.prototxt",
                          modelPath: String = "/Users/guoqiong/intelWork/projects/dlFrames/model/caffe/bvlc_googlenet.caffemodel",
                          folder: String = "/Users/guoqiong/intelWork/projects/dlFrames/data/ILSVRC2012_img_val_100/",
                          batchSize: Int = 8
                         )

object ModelInference {

  def main(args: Array[String]): Unit = {

    val defaultParams = InferenceParam()

    val parser = new OptionParser[InferenceParam]("BigDL Example") {
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

  def run(params: InferenceParam) = {

    val conf = Engine.createSparkConf().setAppName("ModelInference")
      .setMaster("local[8]")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    Engine.init

    Logger.getLogger("org").setLevel(Level.ERROR)
    spark.sparkContext.setLogLevel("ERROR")

    val imagesDF = loadImages(params.folder, params.batchSize, spark.sqlContext)

    val model = Module.loadCaffeModel[Float](params.caffeDefPath, params.modelPath)

    val dlmodel: DLModel[Float] = new DLClassifierModel[Float](
      model, Array(3, 224, 224))
      .setBatchSize(params.batchSize)
      .setFeaturesCol("features")
      .setPredictionCol("predict")

    val tranDF = dlmodel.transform(imagesDF)

    tranDF.select("predict", "imageName").show(5)
  }

  def loadImages(path: String, partitionNum: Int, sqlContext: SQLContext): DataFrame = {

    val paths = LocalImageFiles.readPaths(Paths.get(path), hasLabel = false)
    val valRDD = sqlContext.sparkContext.parallelize(imagesLoad(paths, 256), partitionNum)

    val transf = RowToByteRecords() ->
      BytesToBGRImg() ->
      BGRImgCropper(imageSize, imageSize) ->
      BGRImgNormalizer(testMean, testStd) ->
      BGRImgToImageVector()
    val sizeOfFeature = udf((features: Vector) => features.size)
    val valDF: DataFrame = transformDF(sqlContext.createDataFrame(valRDD), transf)
      .withColumn("featureSize", sizeOfFeature(col("features")))

    valDF
  }
}