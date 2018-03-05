package com.intel.analytics.bigdl.example.DLframes

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dlframes.{DLClassifierModel, DLModel}
import com.intel.analytics.bigdl.example.DLframes.Utils.LocalParams
import com.intel.analytics.bigdl.example.imageclassification.MlUtils._
import com.intel.analytics.bigdl.example.imageclassification.RowToByteRecords
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import scopt.OptionParser

object Utils {

  //TODO delete local path
  case class LocalParams(caffeDefPath: String = "/Users/guoqiong/intelWork/git/caffe/models/bvlc_googlenet/deploy.prototxt",
                         modelPath: String = "/Users/guoqiong/intelWork/projects/dlFrames/model/caffe/bvlc_googlenet.caffemodel",
                         folder: String = "/Users/guoqiong/intelWork/projects/dlFrames/data/ILSVRC2012_img_val_100/",
                         batchSize: Int = 16,
                         nEpochs: Int = 10
                        )

  val defaultParams = LocalParams()

  val parser = new OptionParser[LocalParams]("BigDL Example") {
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
    opt[Int]('e', "nEpochs")
      .text("epoch numbers")
      .action((x, c) => c.copy(nEpochs = x))
  }

  //TODO update with ImageTransfer
  def loadImages(path: String, partitionNum: Int, sqlContext: SQLContext): DataFrame = {

    val paths = LocalImageFiles.readPaths(Paths.get(path), hasLabel = false)
    val valRDD = sqlContext.sparkContext.parallelize(imagesLoad(paths, 256), partitionNum)

    val transf = RowToByteRecords() ->
      BytesToBGRImg(1f) ->
      BGRImgCropper(imageSize, imageSize) ->
      BGRImgNormalizer(123, 117, 104, 1, 1, 1) ->
      BGRImgToImageVector()
    val imagesDF: DataFrame = transformDF(sqlContext.createDataFrame(valRDD), transf)

    imagesDF
  }
}


object ModelInference {

  def main(args: Array[String]): Unit = {

    val defaultParams = LocalParams()
    Utils.parser.parse(args, defaultParams).map { params =>

      val conf = Engine.createSparkConf().setAppName("ModelInference")
      //  .setMaster("local[8]")
      val spark = SparkSession.builder().config(conf).getOrCreate()
      Engine.init

      Logger.getLogger("org").setLevel(Level.ERROR)
      spark.sparkContext.setLogLevel("ERROR")

      val imagesDF = Utils.loadImages(params.folder, params.batchSize, spark.sqlContext)

      imagesDF.show(5)
      imagesDF.printSchema()

      val model = Module.loadCaffeModel[Float](params.caffeDefPath, params.modelPath)

      val dlmodel: DLModel[Float] = new DLClassifierModel[Float](
        model, Array(3, 224, 224))
        .setBatchSize(params.batchSize)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")

      val tranDF = dlmodel.transform(imagesDF)

      tranDF.select("prediction", "imageName").show(5)
    }
  }
}