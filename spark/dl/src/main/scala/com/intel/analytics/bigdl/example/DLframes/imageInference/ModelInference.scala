package com.intel.analytics.bigdl.example.DLframes.imageInference

import java.net.URL
import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dlframes.{DLClassifierModel, DLModel}
import com.intel.analytics.bigdl.example.DLframes.imageInference.Utils.LocalParams
import com.intel.analytics.bigdl.example.imageclassification.MlUtils._
import com.intel.analytics.bigdl.example.imageclassification.RowToByteRecords
import com.intel.analytics.bigdl.example.loadmodel.InceptionPreprocessor.imageSize
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import scopt.OptionParser

import scala.io.Source

object Utils {

  //TODO delete local path
  case class LocalParams(caffeDefPath: String = "/Users/guoqiong/intelWork/git/caffe/models/bvlc_googlenet/deploy.prototxt",
                         modelPath: String = "/Users/guoqiong/intelWork/projects/dlFrames/model/caffe/bvlc_googlenet.caffemodel",
                         folder: String = "/Users/guoqiong/intelWork/projects/dlFrames/data/ILSVRC2012_img_val_20/",
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
      BytesToBGRImg() ->
      BGRImgCropper(imageSize, imageSize) ->
      BGRImgNormalizer(123, 117, 104, 1, 1, 1) ->
      BGRImgToImageVector()
    val imagesDF: DataFrame = transformDF(sqlContext.createDataFrame(valRDD), transf)

    imagesDF
  }

  // Resnet
  def loadImages2(path: String, partitionNum: Int, sqlContext: SQLContext): DataFrame = {

    val paths = LocalImageFiles.readPaths(Paths.get(path), hasLabel = false)
    val valRDD = sqlContext.sparkContext.parallelize(imagesLoad(paths, 256), partitionNum)

    val transf = RowToByteRecords() ->
      BytesToBGRImg() ->
      new BGRImgCropper(imageSize, imageSize, cropperMethod = CropCenter) ->
      BGRImgNormalizer(123, 117, 104, 1, 1, 1) ->
      BGRImgToImageVector()
    val imagesDF: DataFrame = transformDF(sqlContext.createDataFrame(valRDD), transf)

    imagesDF
  }


  def loadImages3(path: String, partitionNum: Int, sqlContext: SQLContext): DataFrame = {

    val imageFrame: ImageFrame = ImageFrame.read(path, sqlContext.sparkContext)
    val transformer = Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(123, 117, 104, 1, 1, 1) -> MatToTensor() -> ImageFrameToSample()
    val transformed: ImageFrame = transformer(imageFrame)
    val imageRDD = transformed.toDistributed().rdd.map { im =>
      (im.uri, im[Sample[Float]](ImageFeature.sample).getData())
    }
    val imageDF = sqlContext.createDataFrame(imageRDD)
      .withColumnRenamed("_1", "imageName")
      .withColumnRenamed("_2", "features")
    imageDF
  }

}

object ModelInference {

  def main(args: Array[String]): Unit = {

    val defaultParams = LocalParams()
    Utils.parser.parse(args, defaultParams).map { params =>

      val conf = Engine.createSparkConf().setAppName("ModelInference")
        .setMaster("local[8]")
      val spark = SparkSession.builder().config(conf).getOrCreate()
      Engine.init

      Logger.getLogger("org").setLevel(Level.ERROR)
      spark.sparkContext.setLogLevel("ERROR")

      val imagesDF = Utils.loadImages3(params.folder, params.batchSize, spark.sqlContext)

      imagesDF.show(10)
      imagesDF.printSchema()

      val model = Module.loadCaffeModel[Float](params.caffeDefPath, params.modelPath)
      val labelFile: URL = getClass().getResource("/imagenet_classname.txt")
      val labelMap = Source.fromURL(labelFile).getLines().zipWithIndex.map(x => (x._2, x._1)).toMap
      val toStringLabel = udf { predict: Int =>
        labelMap(predict - 1)
      }
      //val model = Module.loadModule("/Users/guoqiong/intelWork/projects/dlFrames/model/bigdl/bigdl_inception-v1_imagenet_0.4.0.model")

      val dlmodel: DLModel[Float] = new DLClassifierModel[Float](
        model, Array(3, 224, 224))
        .setBatchSize(params.batchSize)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")

      val tranDF = dlmodel.transform(imagesDF)

      tranDF.select("imageName", "prediction")
        .withColumn("top1Class", toStringLabel(col("prediction")))
        .show(100, false)

    }
  }
}