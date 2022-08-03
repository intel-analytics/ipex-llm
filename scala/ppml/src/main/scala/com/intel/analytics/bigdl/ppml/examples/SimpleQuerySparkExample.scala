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
import org.apache.spark.sql.functions.desc
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

    // next, we will count how many people in each kind of job and their average age
    timing("processing") {
      // load csv file to data frame with ppmlcontext.
      val csvDF = timing("1/4 loadCsv") {
        sc.read(cryptoMode = arguments.inputEncryptMode).option("header", "true")
          .csv(arguments.inputPath + "/people.csv")
      }
      // load parquet file to data frame with ppmlcontext.
      val parquetDF = timing("2/4 loadParquet") {
        sc.read(cryptoMode = arguments.inputEncryptMode)
          .parquet(arguments.inputPath + "/people.parquet")
      }

      val result = timing("3/4 doSQLOperations") {
        // union
        val unionDF = csvDF.unionByName(parquetDF)
        // filter
        val filterDF = unionDF.filter(unionDF("age").between(20, 40))
        // count people in each job
        val countDF = filterDF.groupBy("job").count()
        // calculate average age in each job
        val avgDF = filterDF.groupBy("job")
          .avg("age")
          .withColumnRenamed("avg(age)", "average_age")
        // join and sort
        countDF.join(avgDF, "job").sort(desc("age"))
      }

      result.show()

      // save as a json file
      timing("4/4 encryptAndSaveOutputs") {
        // save data frame using spark kms context
        sc.write(result, cryptoMode = arguments.outputEncryptMode)
          .mode("overwrite")
          .json(arguments.outputPath)
      }
    }
  }
}
