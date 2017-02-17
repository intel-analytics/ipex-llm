
package com.intel.analytics.bigdl.example.udf

import com.intel.analytics.bigdl.nn. Module
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.slf4j.{Logger, LoggerFactory}
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._


import scala.reflect.ClassTag
import scala.util.Random

 /**
  * Created by jwang on 2/14/17.
  */
object TextClassifierUDF {

  val log: Logger = LoggerFactory.getLogger(this.getClass)
  Logger4j.getLogger("org").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("akka").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("breeze").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Levle4j.INFO)

  import Options._

  def main(args: Array[String]): Unit = {

    localParser.parse(args, TextClassificationParams()).map { param =>

      log.info(s"Current parameters: $param")

      val textClassification = new TextClassifier(param)
      val sc = textClassification.getContext()
      val sqlContext = new SQLContext(sc)
      // get train and validation rdds
      val rdds = textClassification.getData(sc)
      // get model
      val localModel = if (param.modelPath.isDefined) {
        Module.load[Float](param.modelPath.get)
      } else {
        val trainedModel = textClassification.train(sc, rdds)
        // after trainning, save model
        if (param.checkpoint.isDefined) {
          trainedModel.save(s"${param.checkpoint.get}/model.1", true)
        } else {
          trainedModel
        }
      }


      // create test dataframe
      val testRDD = rdds(1).mapPartitions(Random.shuffle(_)).map {
        case (filename, text, label) =>
          (filename, new DenseVector(text.flatten.map(_.toDouble)), label.toInt)
      }
      val testDF = sqlContext.createDataFrame(testRDD)

      // add column name to dataframe
      val columnNames = Seq("fileName", "features", "label")
      val df = testDF.toDF(columnNames: _*)
      df.show()

      // broadcast models and sampleshape
      val model = localModel.evaluate()
      val modelBroadCast = sc.broadcast(model)
      val sampleShape = Array(param.maxSequenceLength, param.embeddingDim)
      val sampleShapeBroadCast = sc.broadcast(sampleShape)

      // define udf
      def predict[T: ClassTag](col: DenseVector)
                              (implicit ev: TensorNumeric[T]): Int = {
        val featureTensor: Tensor[T] = Tensor[T]()
        var featureData: Array[T] = null
        val sampleShape = sampleShapeBroadCast.value
        val sampleSize = sampleShape.product
        val localModel = modelBroadCast.value

        // create tensor from input column
        if (featureData == null) {
          featureData = new Array[T](1 * sampleSize)
        }
        Array.copy(col.toArray.map(ev.fromType(_)), 0,
          featureData, 0, sampleSize)

        featureTensor.set(Storage[T](featureData),
          storageOffset = 1, sizes = Array(1) ++ sampleShape)

        val tensorBuffer = featureTensor.transpose(2, 3)

        // predict
        val output = localModel.forward(tensorBuffer).toTensor[T]
        val predict = if (output.dim == 2) {
          output.max(2)._2.squeeze().storage().array()
        } else if (output.dim == 1) {
          output.max(1)._2.squeeze().storage().array()
        } else {
          throw new IllegalArgumentException
        }
        ev.toType[Int](predict(0))
      }

      // declare udf for data frame
      val classiferUDF = udf(predict[Float](_: DenseVector))

      // play with udf in data frame
      import sqlContext.implicits._
      df.withColumn("textType", classiferUDF($"features"))
        .select("fileName", "textType", "label").show(param.showNum)
      df.filter(classiferUDF($"features") === 1)
        .select("fileName", "label").show(param.showNum)
      df.withColumn("textType", classiferUDF($"features"))
        .filter("textType = 1")
        .select("fileName", "textType", "label").show(param.showNum)

      // register udf in sql context
      sqlContext.udf.register("textClassifier", predict[Float] _)
      // play with udf in sqlcontext
      df.registerTempTable("textTable")
      df.filter("textClassifier(features) = 1").select("fileName", "label").show(param.showNum)
      sqlContext.sql("select fileName, textClassifier(features) as textType, label from textTable")
        .show(param.showNum)
      sqlContext
        .sql("select fileName, textClassifier(features) as textType, label " +
          "from textTable where textClassifier(features) = 1")
        .show(param.showNum)

      sc.stop()
    }
  }
}
