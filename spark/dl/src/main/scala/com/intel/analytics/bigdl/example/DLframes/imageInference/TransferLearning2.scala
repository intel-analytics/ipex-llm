package com.intel.analytics.bigdl.example.DLframes.imageInference

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.dlframes.DLModel
import com.intel.analytics.bigdl.example.DLframes.imageInference.Utils.LocalParams
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.ml.adapter.{HasFeaturesCol, HasPredictionCol, SchemaUtils}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{ParamMap, ParamPair, Params}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.reflect.ClassTag

object TransferLearning2 {

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
      val imagesDF: DataFrame = Utils.loadImages3(params.folder, params.batchSize, spark.sqlContext)
        .withColumn("label", createLabel(col("imageName")))
        .withColumnRenamed("features", "imageFeatures")

      imagesDF.show(10)
      imagesDF.printSchema()

      val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.90, 0.10), seed = 1L)

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

      val dl = new DLModel[Float](
        loaded, Array(3, 224, 224))
        .setBatchSize(params.batchSize)
        .setFeaturesCol("imageFeatures")
        .setPredictionCol("xy")

      val tmp = dl.transform(trainingDF)

      println("-----------after dl model --------------------")

      tmp.printSchema()

      val featurizer = new Array2Vector()
        .setFeaturesCol("xy")
        .setPredictionCol("xyz")


      val tmp2: DataFrame = featurizer.transform(tmp)

      println("---------------after featurizer model ---------------")

      tmp2.show(5)
      tmp2.printSchema()

      val sizeOfFeature = udf((features: Vector) => features.size)


      val lr = new LogisticRegression()
        .setMaxIter(20)
        .setRegParam(0.05)
        .setElasticNetParam(0.3)
        .setFeaturesCol("xyz")

      val pipeline = new Pipeline().setStages(
        Array(dl, featurizer, lr))

      val tmp3: LogisticRegressionModel = lr.fit(tmp2)
      val tmp4 = tmp3.transform(tmp2)

      println("---------------------after lr model")
      tmp4.printSchema()

      tmp4.select("imageName", "rawPrediction", "probability", "label", "prediction")
        .show(100, false)

      val pipelineModel = pipeline.fit(trainingDF)

      val time2 = System.nanoTime()

      val count1 = trainingDF.count().toInt
      val count2 = validationDF.count().toInt


      val predictions = pipelineModel.transform(trainingDF)

      predictions.show(200)
      predictions.printSchema()

      val evaluation = new MulticlassClassificationEvaluator().setPredictionCol("prediction")
        .setMetricName("accuracy").evaluate(predictions)
      println("evaluation result on validationDF: " + evaluation)

    }
  }

}

//DL -> Featurizer-> lr
//Featurizer -> Array2Vecor - > lr

//class Featurizer[@specialized(Float, Double) T: ClassTag](@transient val model2: Module[T],
//                                                          var featureSize2: Array[Int],
//                                                          override val uid: String = "DLModel"
//                                                         )(implicit ev: TensorNumeric[T]) extends DLModel(model2, featureSize2, uid) {
//
//  override def transform(dataset: Dataset[_]): DataFrame = {
//
//
//    val df = dataset.toDF()
//    println("---------------inside transform ")
//
//    df.printSchema()
//    val toVector = udf { features: Seq[Double] =>
//      Vectors.dense(features.toArray)
//    }
//
//    df.withColumn(getPredictionCol, toVector(col(getFeaturesCol)))
//  }
//
//  override def transformSchema(schema: StructType): StructType = {
//    schema.add(getPredictionCol, VectorType)
//  }
//}

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
    copyValues(array,extra)
  }
}

