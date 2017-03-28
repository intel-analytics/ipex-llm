/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.example.udfpredictor

import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}
import org.apache.spark.sql.functions._
import org.slf4j.{Logger, LoggerFactory}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType


object  StructuredStreamPredictor {

  private val log: Logger = LoggerFactory.getLogger(this.getClass)
  Logger4j.getLogger("org").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("akka").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("breeze").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Levle4j.INFO)

  import Options._

  def main(args: Array[String]): Unit = {

    localParser.parse(args, TextClassificationParams()).foreach { param =>

      log.info(s"Current parameters: $param")

      // Create spark session
      val sparkConf = Engine.createSparkConf()
      sparkConf.setAppName("Text classification")
        .set("spark.akka.frameSize", 64.toString)
      val spark = SparkSession
        .builder
        .config(sparkConf)
        .getOrCreate()
      Engine.init
      val sc = spark.sparkContext

      val (model, word2Vec, sampleShape) = Utils.getModel(sc, param)

      val predict = Utils.genUdf(sc, model, sampleShape, word2Vec)

      // register udf for data frame
      val classifierUDF = udf(predict)

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
        .csv(Utils.getResourcePath("/example/udfpredictor/types"))

      import spark.implicits._

      val classifyDF1 = df.withColumn("textLabel", classifierUDF($"text"))
        .select("fileName", "text", "textLabel")
      val classifyQuery1 = classifyDF1.writeStream
        .format("console")
        .start()

      val df_join = classifyDF1.join(types, "textLabel")
      val classifyQuery_join = df_join.writeStream
        .format("console")
        .start()

      val filteredDF1 = df.filter(classifierUDF($"text") === 8)
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
      spark.udf.register("textClassifier", predict)
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
