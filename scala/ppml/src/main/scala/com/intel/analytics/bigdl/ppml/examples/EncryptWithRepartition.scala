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
import org.apache.spark.{SparkConf}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.slf4j.LoggerFactory

object EncryptWithRepartition extends Supportive {

  def main(args: Array[String]): Unit = {
    val logger = LoggerFactory.getLogger(getClass)

    // parse parameter
    val arguments = EncryptIOArguments.parser.parse(args, EncryptIOArguments()) match {
        case Some(arguments) =>
          logger.info(s"starting with $arguments"); arguments
        case None =>
          EncryptIOArguments.parser.failure("miss args, please see the usage info"); null
      }

    val sparkConf = new SparkConf().setMaster("local[4]")
    val sc = PPMLContext.initPPMLContext(sparkConf, "EncryptWithRepartition", arguments.ppmlArgs())

    timing("processing") {
      // load csv file to data frame with ppmlcontext.
      val df = timing("1/2 load Inputs and Repartition") {
        sc.read(cryptoMode = arguments.inputEncryptMode).option("header", "true")
          .csv(arguments.inputPath).repartition(arguments.outputPartitionNum)
      }

      timing("2/2 encryptAndSaveOutputs") {
        // save data frame using spark kms context
        sc.write(df, cryptoMode = arguments.outputEncryptMode).mode("overwrite")
          .option("header", true).csv(arguments.outputPath)
      }
    }
  }
}
