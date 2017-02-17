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

import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import com.intel.analytics.bigdl.example.utils.WordMeta
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType
import org.apache.log4j.{Level, Logger}

object  StructuredStreamPredictor {

  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.example").setLevel(Level.INFO)

  //  import Options._

  def main(args: Array[String]): Unit = {

    Utils.localParser.parse(args, TextClassificationUDFParams()).foreach { param =>

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

      var word2Meta = None: Option[Map[String, WordMeta]]
      var word2Index = None: Option[Map[String, Int]]
      var word2Vec = None: Option[Map[Float, Array[Float]]]

      val result = Utils.getModel(sc, param)

      val model = result._1
      word2Meta = result._2
      word2Vec = result._3
      val sampleShape = result._4

      // if not train, load word meta from file
      if (word2Meta.isEmpty) {
        val word2IndexMap = sc.textFile(s"${param.baseDir}/word2Meta.txt").map(item => {
          val tuple = item.stripPrefix("(").stripSuffix(")").split(",")
          (tuple(0), tuple(1).toInt)
        }).collect()
        word2Index = Some(word2IndexMap.toMap)
      } else {
        // already trained, use existing word meta
        val word2IndexMap = collection.mutable.HashMap.empty[String, Int]
        for((word, wordMeta) <- word2Meta.get) {
          word2IndexMap += (word -> wordMeta.index)
        }
        word2Index = Some(word2IndexMap.toMap)
      }

      // if not train, create word vec
      if (word2Vec.isEmpty) {
        word2Vec = Some(Utils.getWord2Vec(word2Index.get))
      }
      val predict = Utils.genUdf(sc, model, sampleShape, word2Index.get, word2Vec.get)

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
