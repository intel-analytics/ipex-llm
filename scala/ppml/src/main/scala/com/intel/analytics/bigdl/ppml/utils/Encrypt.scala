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

package com.intel.analytics.bigdl.ppml.utils

import com.intel.analytics.bigdl.ppml.utils.Supportive
import com.intel.analytics.bigdl.ppml.PPMLContext
import org.apache.spark.sql.SparkSession
import com.intel.analytics.bigdl.ppml.crypto.{CryptoMode, PLAIN_TEXT}

object Encrypt extends Supportive {

  def main(args: Array[String]): Unit = {
    val arguments = argumentsParser.parse(args, EncryptArguments()) match {
        case Some(arguments) => arguments
        case None => argumentsParser.failure("miss args, please see the usage info"); null
    }
    val (inputDataSourcePath, outputDataSinkPath, cryptoMode, dataSourceType, action) = (
        arguments.inputDataSourcePath,
        arguments.outputDataSinkPath,
        CryptoMode.parse(arguments.cryptoMode),
        arguments.dataSourceType,
        arguments.action
    )

    val sparkSession: SparkSession = SparkSession.builder().getOrCreate()
    val sc: PPMLContext = PPMLContext.initPPMLContext(sparkSession)
    if (action.equals("encrypt")) {
      dataSourceType match {
        case "csv" =>
          val df = sc.read(PLAIN_TEXT).csv(inputDataSourcePath)
          sc.write(df, cryptoMode).csv(outputDataSinkPath)
        case "json" =>
          val df = sc.read(PLAIN_TEXT).json(inputDataSourcePath)
          sc.write(df, cryptoMode).json(outputDataSinkPath)
        case "parquet" =>
          val df = sc.read(PLAIN_TEXT).parquet(inputDataSourcePath)
          sc.write(df, cryptoMode).parquet(outputDataSinkPath)
        case "textFile" =>
          import sparkSession.implicits._
          val df = sc.textFile(inputDataSourcePath).toDF
          sc.write(df, cryptoMode).text(outputDataSinkPath)
        case _ =>
          argumentsParser.failure("wrong dataSourceType, please see the usage info")
      }
    } else if (action.equals("decrypt")) {
      dataSourceType match {
        case "csv" =>
          val df = sc.read(cryptoMode).csv(inputDataSourcePath)
          sc.write(df, PLAIN_TEXT).csv(outputDataSinkPath)
        case "json" =>
          val df = sc.read(cryptoMode).json(inputDataSourcePath)
          sc.write(df, PLAIN_TEXT).json(outputDataSinkPath)
        case "parquet" =>
          val df = sc.read(cryptoMode).parquet(inputDataSourcePath)
          sc.write(df, PLAIN_TEXT).parquet(outputDataSinkPath)
        case "textFile" =>
          import sparkSession.implicits._
          val df = sc.textFile(inputDataSourcePath, cryptoMode = cryptoMode).toDF
          sc.write(df, PLAIN_TEXT).text(outputDataSinkPath)
        case _ =>
          argumentsParser.failure("wrong dataSourceType, please see the usage info")
      }
    } else {
      argumentsParser.failure("wrong action, please see the usage info")
    }
  }

  val argumentsParser =
    new scopt.OptionParser[EncryptArguments]("PPML Encrypt Arguments") {
      opt[String]('i', "inputDataSourcePath")
        .action((x, c) => c.copy(inputDataSourcePath = x))
        .text("path of input data to encrypt e.g. file://... or hdfs://...")
      opt[String]('o', "outputDataSinkPath")
        .action((x, c) => c.copy(outputDataSinkPath = x))
        .text("output path to save encrypted data e.g. file://... or hdfs://...")
      opt[String]('m', "cryptoMode")
        .action((x, c) => c.copy(cryptoMode = x))
        .text("encryption mode, aes/cbc/pkcs5padding for csv, json and textFile,"
          + " and aes_gcm_v1 or aes_gcm_ctr_v1 for parquet")
      opt[String]('t', "dataSourceType")
        .action((x, c) => c.copy(dataSourceType = x))
        .text("file type of input data source, csv, json, parquet or textFile")
      opt[String]('a', "action")
        .action((x, c) => c.copy(dataSourceType = x))
        .text("action type of encrypt or decrypt file, default is encrypt")
    }
}

case class EncryptArguments(inputDataSourcePath: String = "input_data_path",
                            outputDataSinkPath: String = "output_save_path",
                            cryptoMode: String = "aes/cbc/pkcs5padding",
                            dataSourceType: String = "csv",
                            action: String = "encrypt")


