/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.example.structuredStreamUdf

import java.io.{File, InputStream, PrintWriter}

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}
import org.apache.spark.sql.functions._
import org.slf4j.{Logger, LoggerFactory}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType

import scala.reflect.ClassTag

object TextClassifierConsumerParquet {

  val log: Logger = LoggerFactory.getLogger(this.getClass)
  Logger4j.getLogger("org").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("akka").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("breeze").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Levle4j.INFO)

  import Options._

   /**
     * Get resource file path
     * @param resource
     * @return file path
     */
   def getResourcePath(resource: String) : String = {
     val stream : InputStream = getClass.getResourceAsStream(resource)
     val lines = scala.io.Source.fromInputStream( stream ).mkString
     val file = File.createTempFile(resource, "")
     val pw = new PrintWriter(file)
     pw.write(lines)
     pw.close
     file.getAbsolutePath
   }

  def main(args: Array[String]): Unit = {

    localParser.parse(args, TextClassificationParams()).map { param =>

      log.info(s"Current parameters: $param")

      val textClassification = new TextClassifier(param)

      // Create spark session
      val spark = SparkSession
        .builder
        .config(Engine.init(param.nodeNum, param.coreNum, true).get
          .setAppName("Text classification")
          .set("spark.akka.frameSize", 64.toString)
          .set("spark.task.maxFailures", "1"))
        .appName("StructuredStreamingUdf")
        .getOrCreate()
      val sc = spark.sparkContext

      // get model
      val localModel = if (param.modelPath.isDefined) {
        Module.load[Float](param.modelPath.get)
      } else {
        // get train and validation rdds
        val rdds = textClassification.getData(sc)
        // train
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

      // register udf for data frame
      val classiferUDF = udf(predict[Float](_: String))

      val textSchema = new StructType().add("filename", "string").add("text", "string")
      // stream dataframe
      val df = spark.readStream
        .schema(textSchema)
        .parquet(param.testDir)

      val typeSchema = new StructType().add("textType", "string").add("textLabel", "string")
      // static dataframe
      val types = spark.read
        .format("csv")
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .schema(typeSchema)
        .csv(getResourcePath("/types.csv"))

      import spark.implicits._

      val classifyDF1 = df.withColumn("textLabel", classiferUDF($"text"))
        .select("fileName", "text", "textLabel")
      val classifyQuery1 = classifyDF1.writeStream
        .format("console")
        .start()

      val df_join = classifyDF1.join(types, "textLabel")
      val classifyQuery_join = df_join.writeStream
        .format("console")
        .start()

      val filteredDF1 = df.filter(classiferUDF($"text") === 8)
      val filteredQuery1 = filteredDF1.writeStream
        .format("console")
        .start()

      // aggregation
      val typeCount = classifyDF1.groupBy($"textLabel").count()
      val aggQuery = typeCount.writeStream
        .outputMode("complete")
        .format("console")
        .start()

      // play with udf in sqlcontext
      spark.udf.register("textClassifier", predict[Float] _)
      df.createOrReplaceTempView("textTable")

      val classifyDF2 = spark
        .sql("SELECT fileName, textClassifier(text) AS textType_sql, text FROM textTable")
      val classifyQuery2 = classifyDF2.writeStream
        .format("console")
        .start()

      val filteredDF2 = spark
        .sql("SELECT fileName, textClassifier(text) AS textType_sql, text " +
          "FROM textTable WHERE textClassifier(text) = 9")
      val filteredQuery2 = filteredDF2.writeStream
        .format("console")
        .start()

      classifyQuery1.awaitTermination()
      classifyQuery_join.awaitTermination()
      filteredQuery1.awaitTermination()
      aggQuery.awaitTermination()
      classifyQuery2.awaitTermination()
      filteredQuery2.awaitTermination()
    }
  }
}
