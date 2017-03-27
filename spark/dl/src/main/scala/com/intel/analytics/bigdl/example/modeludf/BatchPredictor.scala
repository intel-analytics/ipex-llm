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
package com.intel.analytics.bigdl.example.modeludf

import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.slf4j.{Logger, LoggerFactory}
import org.apache.spark.sql.SQLContext

object BatchPredictor {

  private val log: Logger = LoggerFactory.getLogger(this.getClass)
  Logger4j.getLogger("org").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("akka").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("breeze").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Levle4j.INFO)

  import Options._

  def main(args: Array[String]): Unit = {

    localParser.parse(args, TextClassificationParams()).foreach { param =>

      log.info(s"Current parameters: $param")

      val conf = Engine.createSparkConf()
      conf.setAppName("Text classification")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      // Create spark session
      val spark = new SQLContext(sc)
      import spark.implicits._

      val (model, word2Vec, sampleShape) = Utils.getModel(sc, param)

      val predict = Utils.genUdf(sc, model, sampleShape, word2Vec)

      // register udf for data frame
      val classifierUDF = udf(predict)

      val data = Utils.loadTestData(param.testDir)

      val df = spark.createDataFrame(data)

      // static dataframe
      val types = sc.textFile(Utils.getResourcePath("/types"))
        .filter(!_.contains("textType"))
        .map { line =>
          val words = line.split(",")
          (words(0).trim, words(1).trim.toInt)
        }.toDF("textType", "textLabel")

      val classifyDF1 = df.withColumn("textLabel", classifierUDF($"text"))
        .select("filename", "text", "textLabel").orderBy("filename")
      classifyDF1.show()

      val filteredDF1 = df.filter(classifierUDF($"text") === 8).orderBy("filename")
      filteredDF1.show()

      val df_join = classifyDF1.join(types, "textLabel").orderBy("filename")
      df_join.show()

      // aggregation
      val typeCount = classifyDF1.groupBy($"textLabel").count().orderBy("textLabel")
      typeCount.show()

      // play with udf in sqlcontext
      spark.udf.register("textClassifier", predict)
      df.registerTempTable("textTable")

      val classifyDF2 = spark
        .sql("SELECT filename, textClassifier(text) AS textType_sql, text " +
          "FROM textTable order by filename")
      classifyDF2.show()

      val filteredDF2 = spark
        .sql("SELECT filename, textClassifier(text) AS textType_sql, text " +
          "FROM textTable WHERE textClassifier(text) = 9 order by filename")
      filteredDF2.show()
    }

  }

}
