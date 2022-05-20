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
import com.intel.analytics.bigdl.ppml.utils.{EncryptIOArguments, KeyReaderWriter}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.slf4j.LoggerFactory

object SimpleQuerySparkExample {

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

    // load csv file to data frame with ppmlcontext.
    val df = sc.read(cryptoMode = arguments.inputEncryptMode).option("header", "true")
      .csv(arguments.inputPath + "/people.csv")

    // Select only the "name" column
    df.select("name").count()

    // Select everybody, but increment the age by 1
    df.select(df("name"), df("age") + 1).show()

    // Select Developer and records count
    val developers = df.filter(df("job") === "Developer" and df("age").between(20, 40)).toDF()
    developers.count()

    Map[String, DataFrame]({
      "developers" -> developers
    })

    // save data frame using spark kms context
    sc.write(developers, cryptoMode = arguments.outputEncryptMode).mode("overwrite")
      .option("header", true).csv(arguments.outputPath)

    // sava data frame to parquet
    // writeParquet(arguments, sc, df)
  }

  def writeParquet(arguments: EncryptIOArguments, sc: PPMLContext, df: DataFrame): Unit = {
    val keyReaderWriter = new KeyReaderWriter
    val primaryKeyPlaintext = keyReaderWriter.readKeyFromFile(arguments.primaryKeyPath)
    val sparkContext = sc.getContext
    sparkContext.hadoopConfiguration.set("parquet.encryption.kms.client.class" ,
      "com.intel.analytics.bigdl.ppml.utils.InMemoryKMS")
    sparkContext.hadoopConfiguration.set("parquet.encryption.key.list" ,
      "primaryKey:" + primaryKeyPlaintext)
    sparkContext.hadoopConfiguration.set("parquet.crypto.factory.class" ,
      "org.apache.parquet.crypto.keytools.PropertiesDrivenCryptoFactory")
    // get the column need to encrypt
    val schema = df.schema
    var encryptColumn = ""
    var fieldNum = 0
    schema.foreach(field => {
      if(fieldNum == 0) {
        encryptColumn += ":"
      } else {
        encryptColumn += ","
      }
      encryptColumn += field.name
      fieldNum += 1
    })
    df.write.
      option("parquet.encryption.column.keys" , "primaryKey"+encryptColumn).
      option("parquet.encryption.footer.key" , "primaryKey").
      parquet("table.parquet.encrypted")
  }
}
