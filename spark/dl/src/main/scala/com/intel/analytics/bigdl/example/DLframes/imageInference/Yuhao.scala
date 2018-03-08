package com.intel.analytics.bigdl.example.DLframes.imageInference

import java.net.URL

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame, ImageFrameToSample, MatToTensor}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.DLClassifierModel
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

import scala.io.Source

/**
  * Created by yuhao on 2/15/18.
  */
object ImageFrameTest {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val conf = Engine.createSparkConf()
      .setMaster("local[1]")
      .setAppName("DLClassifierLogisticRegression")
    val sc = new SparkContext(conf)
    Engine.init

    val imagePath = "/Users/guoqiong/intelWork/projects/dlFrames/data/ILSVRC2012_img_val_20/"
    val imageFrame = ImageFrame.read(imagePath, sc)
    val fi = imageFrame.toDistributed().rdd.first()
    val transformer = Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(123, 117, 104, 1, 1, 1) -> MatToTensor() -> ImageFrameToSample()
    val transformed = transformer(imageFrame)

    val imf = transformed.toDistributed().rdd.take(1).head

    val model = Module.loadModule("/Users/guoqiong/intelWork/projects/dlFrames/model/bigdl/bigdl_inception-v1_imagenet_0.4.0.model")

    val labelFile: URL = getClass().getResource("/imagenet_classname.txt")
    val labelMap = Source.fromURL(labelFile).getLines().zipWithIndex.map(x => (x._2, x._1)).toMap

    val output = model.predictImage(transformed)
    output.toDistributed().rdd.collect().sortBy(_.uri()).foreach { imageFeature =>
      //      print(imageFeature.uri() + ": ")
      //      println(imageFeature(ImageFeature.predict))
      //      val predictDist = imageFeature(ImageFeature.predict).asInstanceOf[Tensor[Float]].toArray()

      //      println(predictDist.take(10).mkString(", "))
      //      val predict = predictDist.zipWithIndex.maxBy(_._1)._2
      //      println(predict + "; " + labelMap(predict))
    }


    {
      val imageRDD = transformed.toDistributed().rdd.map { im =>
        (im.uri(), im[Sample[Float]](ImageFeature.sample).getData())
      }
      val spark = SQLContext.getOrCreate(sc)
      val imageDF = spark.createDataFrame(imageRDD)
        .withColumnRenamed("_1", "path")
        .withColumnRenamed("_2", "features")

      val dlModel = new DLClassifierModel(model, Array(3, 224, 224)).setBatchSize(4)
      val toStringLabel = udf { predict: Int =>
        labelMap(predict - 1)
      }
      dlModel.transform(imageDF).orderBy("path").select("path", "prediction")
        .withColumn("top1Class", toStringLabel(col("prediction")))
        .show(100, false)
    }
  }

}
