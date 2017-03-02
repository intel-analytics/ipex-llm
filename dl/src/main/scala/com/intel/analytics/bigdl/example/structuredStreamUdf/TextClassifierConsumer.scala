
package com.intel.analytics.bigdl.example.structuredStreamUdf

import com.intel.analytics.bigdl.example.structuredStreamUdf.TextProducerKafka.Sample
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Engine
import kafka.serializer.StringDecoder
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}

import org.apache.spark.sql.functions._
import org.slf4j.{Logger, LoggerFactory}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType

import scala.reflect.ClassTag

/**
  * Created by jwang on 2/14/17.
  */
object TextClassifierConsumer {

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
      //     val sc = textClassification.getContext()

      val spark = SparkSession
        .builder
        .config(Engine.init(param.nodeNum, param.coreNum, true).get
          .setAppName("Text classification")
          .set("spark.akka.frameSize", 64.toString)
          .set("spark.task.maxFailures", "1"))
        .appName("StructuredStreamingUdf")
        .getOrCreate()
      val sc = spark.sparkContext
      //     val sqlContext = new SQLContext(sc)
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

      // broadcast models and sample shape
      val model = localModel.evaluate()
      val modelBroadCast = sc.broadcast(model)
      val sampleShape = Array(param.maxSequenceLength, param.embeddingDim)
      val sampleShapeBroadCast = sc.broadcast(sampleShape)
      val word2Vec = textClassification.buildWord2VecMap()
      val word2VecBroadcast = sc.broadcast(word2Vec)


      // define udf
      def predict[T: ClassTag](text: String)
                              (implicit ev: TensorNumeric[T]): Int = {

        val sampleShape = sampleShapeBroadCast.value
        val sequenceLen = sampleShape(0)
        val embeddingDim = sampleShape(1)
        // first to tokens
        val tokens = text.replaceAll("[^a-zA-Z]", " ").toLowerCase().split("\\s+").filter(_.size > 2)
        // shaping
        val paddedTokens = if (tokens.length > sequenceLen) {

          tokens.slice(tokens.length - sequenceLen, tokens.length)

        } else {
          tokens ++ Array.fill[String](sequenceLen - tokens.length)("invalidword")
        }
        // to vectors
        val word2Vec = word2VecBroadcast.value
        val data = paddedTokens.map { word: String =>
          if (word2Vec.contains(word)) {
            word2Vec(word)
          } else {
            // Treat it as zeros if cannot be found from pre-trained word2Vec
            Array.fill[Float](embeddingDim)(0)
          }
        }.flatten


        val featureTensor: Tensor[T] = Tensor[T]()
        var featureData: Array[T] = null

        val sampleSize = sampleShape.product
        val localModel = modelBroadCast.value

        // create tensor from input column
        if (featureData == null) {
          featureData = new Array[T](1 * sampleSize)
        }
        Array.copy(data.map(ev.fromType(_)), 0,
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

      val classiferUDF = udf(predict[Float](_: String))

      val textSchema = new StructType().add("filename", "string").add("text", "string")
      val df = spark.readStream
        .schema(textSchema)
        .parquet(param.testDir)

      import spark.implicits._
      //      println(df.isStreaming)
      val ds = df.as[Sample]

      // no aggregation
      val newDF = df.withColumn("textType", classiferUDF($"text"))
        .select("fileName", "textType", "text")

      val noAggQuery = newDF.writeStream
        .format("console")
        .start()

      // aggregation
      val typeCount = newDF.groupBy($"textType").count()

      val aggQuery = typeCount.writeStream
        .outputMode("complete")
        .format("console")
        .start()

      noAggQuery.awaitTermination()
      aggQuery.awaitTermination()
      //      ds1.filter(classiferUDF($"text") === 1)
      //        .show(param.showNum)
      //      ds1.withColumn("textType", classiferUDF($"text"))
      //        .select("fileName", "text", "textType")
      //        .filter("textType = 1")
      //        .show(param.showNum)

      // register udf in sql context
      //      spark.udf.register("myUDF", (arg1: Int, arg2: String) => arg2 + arg1)
      //      spark.udf.register("textClassifier", predict[Float] _)

      // play with udf in sqlcontext
      //      ds1.registerTempTable("textTable")
      //      ds1.createOrReplaceTempView("textTable");
      //      ds1.filter("textClassifier(text) = 1").select("fileName", "text").show(param.showNum)
      //      spark.sql("SELECT fileName, textClassifier(text) AS textType, text FROM textTable")
      //        .show(param.showNum)
      //      spark
      //        .sql("SELECT fileName, textClassifier(text) AS textType, text " +
      //          "FROM textTable WHERE textClassifier(text) = 1")
      //        .show(param.showNum)

      //      sc.stop()
    }
  }
}
