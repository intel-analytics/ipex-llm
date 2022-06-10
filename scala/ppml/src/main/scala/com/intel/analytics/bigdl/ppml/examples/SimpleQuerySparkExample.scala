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

package com.intel.analytics.bigdl.ppml.examples

import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.kms.{EHSMKeyManagementService, KMS_CONVENTION, SimpleKeyManagementService}
import com.intel.analytics.bigdl.ppml.utils.{EncryptIOArguments, Supportive}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.slf4j.LoggerFactory

object SimpleQuerySparkExample extends Supportive {

  def main(args: Array[String]): Unit = {
    val logger = LoggerFactory.getLogger(getClass)

    // parse parameter
    val arguments = EncryptIOArguments.parser.parse(args, EncryptIOArguments()) match {
        case Some(arguments) =>
          logger.info(s"starting with $arguments"); arguments
        case None =>
          EncryptIOArguments.parser.failure("miss args, please see the usage info"); null
      }

    val sc = PPMLContext.initPPMLContext("SimpleQuery", arguments.ppmlArgs())

    // read kms args from spark-defaults.conf
    // val sc = PPMLContext.initPPMLContext("SimpleQuery")

    timing("processing") {
      // load csv file to data frame with ppmlcontext.
      val df = timing("1/3 loadInputs") {
        sc.read(cryptoMode = arguments.inputEncryptMode).option("header", "true")
          .csv(arguments.inputPath + "/people.csv")
      }

      val developers = timing("2/3 doSQLOperations") {
        // Select only the "name" column
        df.select("name").count()

        // Select everybody, but increment the age by 1
        df.select(df("name"), df("age") + 1).show()

      // Select Developer and records count
        val developers = df.filter(df("job") === "Developer" and df("age").between(20, 40)).toDF()
        developers.count()

        developers
      }

      // Map[String, DataFrame]({
      //  "developers" -> developers
      // })

      timing("3/3 encryptAndSaveOutputs") {
        // save data frame using spark kms context
        sc.write(developers, cryptoMode = arguments.outputEncryptMode).mode("overwrite")
          .option("header", true).csv(arguments.outputPath)
      }
    }
  }
}
