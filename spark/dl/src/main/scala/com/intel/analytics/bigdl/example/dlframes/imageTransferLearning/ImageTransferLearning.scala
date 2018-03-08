package com.intel.analytics.bigdl.example.dlframes.imageTransferLearning

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.dlframes.DLModel
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame, ImageFrameToSample, MatToTensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.adapter.{HasFeaturesCol, HasPredictionCol}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext, SparkSession}
import scopt.OptionParser
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat


object TransferLearning {

  def main(args: Array[String]): Unit = {

    val defaultParams = Utils.LocalParams()

    Utils.parser.parse(args, defaultParams).map { params =>

      val conf = Engine.createSparkConf().setAppName("TransferLearning")
      val spark = SparkSession.builder().config(conf).getOrCreate()
      Engine.init

      val createLabel = udf((name: String) => if (name.contains("cat")) 1.0 else 0.0)
      val imagesDF: DataFrame = Utils.loadImages(params.folder, params.batchSize, spark.sqlContext)
        .withColumn("label", createLabel(col("imageName")))
        .withColumnRenamed("features", "imageFeatures")

      val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.90, 0.10), seed = 1L)

      val criterion = ClassNLLCriterion[Float]()

      val loadedModel: AbstractModule[Activity, Activity, Float] = Module
        .loadCaffeModel[Float](params.caffeDefPath, params.modelPath)

      val featurizer = new DLModel[Float](loadedModel, Array(3, 224, 224))
        .setBatchSize(params.batchSize)
        .setFeaturesCol("imageFeatures")
        .setPredictionCol("tmp1")


      val schemaTransformer = new Array2Vector()
        .setFeaturesCol("tmp1")
        .setPredictionCol("features")

      val lr = new LogisticRegression()
        .setMaxIter(20)
        .setRegParam(0.05)
        .setElasticNetParam(0.3)
        .setFeaturesCol("features")

      val pipeline = new Pipeline().setStages(
        Array(featurizer, schemaTransformer, lr))

      val pipelineModel = pipeline.fit(trainingDF)

      val predictions = pipelineModel.transform(trainingDF)

      predictions.show(200)
      predictions.printSchema()

      val evaluation = new MulticlassClassificationEvaluator().setPredictionCol("prediction")
        .setMetricName("accuracy").evaluate(predictions)
      println("evaluation result on validationDF: " + evaluation)

    }
  }

}

class Array2Vector(override val uid: String = "array2vector") extends Transformer with HasFeaturesCol
  with HasPredictionCol {

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()
    df.printSchema()
    val toVector = udf { features: Seq[Double] =>
      Vectors.dense(features.toArray)
    }

    df.withColumn(getPredictionCol, toVector(col(getFeaturesCol)))
  }

  override def transformSchema(schema: StructType): StructType = {
    schema.add(getPredictionCol, VectorType)
  }

  override def copy(extra: ParamMap): Array2Vector = {
    val array = new Array2Vector()
    copyValues(array, extra)
  }
}

object Utils {

  case class LocalParams(caffeDefPath: String = " ",
                         modelPath: String = " ",
                         folder: String = " ",
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

  def loadImages(path: String, partitionNum: Int, sqlContext: SQLContext): DataFrame = {

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
