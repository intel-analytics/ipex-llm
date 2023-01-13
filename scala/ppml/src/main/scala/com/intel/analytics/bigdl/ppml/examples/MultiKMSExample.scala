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

import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.kms.{EHSMKeyManagementService, KMS_CONVENTION, SimpleKeyManagementService}
import com.intel.analytics.bigdl.ppml.utils.{EncryptIOArguments, Supportive}
import com.intel.analytics.bigdl.ppml.crypto.{CryptoMode, EncryptRuntimeException, PLAIN_TEXT}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.slf4j.LoggerFactory

object MultiKMSExample extends Supportive {
  def main(args: Array[String]): Unit = {
    // get spark session and make ppml context
    val sparkSession = SparkSession.builder().getOrCreate
    val conf = sparkSession.sparkContext.getConf
    val sc = PPMLContext.initPPMLContextMultiKMS(sparkSession)

    //
    timing("processing") {
      // load csv file to data frame with ppmlcontext.
      val simpleKMSDf = timing("1/3 loadInputs") {
        sc.read(cryptoMode = CryptoMode.parse(
          conf.get("spark.bigdl.kms.datasource1.inputEncryptMode")))
          .option("header", "true")
          .csv(conf.get("spark.bigdl.kms.datasource1.inputpath"))
      }

      val ehsmDf = timing("1/3 read data source 2") {
        sc.read(cryptoMode = CryptoMode.parse(
          conf.get("spark.bigdl.kms.datasource2.inputEncryptMode")))
          .option("header", "true")
          .csv(conf.get("spark.bigdl.kms.datasource2.inputpath"))
      }


      val simpleKMSDevelopers = timing("2/3 doSQLOperations") {
        // Select only the "name" column
        simpleKMSDf.select("name").count()

        // Select everybody, but increment the age by 1
        simpleKMSDf.select(simpleKMSDf("name"), simpleKMSDf("age") + 1).show()

      // Select Developer and records count
        val simpleKMSDevelopers = simpleKMSDf.filter(simpleKMSDf("job") === "Developer" and simpleKMSDf("age").between(20, 40)).toDF()
        simpleKMSDevelopers.count()

        simpleKMSDevelopers
      }

      val ehsmDevelopers = timing("2/3 datasource2 do SQL") {
        ehsmDf.select("name").count
        ehsmDf.select(ehsmDf("name"), ehsmDf("age") ).show()
        val ehsmDevelopers = ehsmDf.filter(ehsmDf("job") === "Developer" and ehsmDf("age")
          .between(20, 40)).toDF()
        ehsmDevelopers.count()
        ehsmDevelopers
      }


      timing("3/3 encryptAndSaveOutputs") {
        // save data frame using spark kms context
        Log4Error.invalidInputError(conf.contains("spark.bigdl.kms.datasource1.outputEncryptMode"),
          "output encryput mode not found")

        // write encrypted data
        var simpleKMSOutputPath : String =""
        if (conf.contains("spark.bigdl.kms.datasource1.outputpath")){
          simpleKMSOutputPath = conf.get("spark.bigdl.kms.datasource1.outputpath")
        } else {
          simpleKMSOutputPath = conf.get("spark.bigdl.kms.datasource1.inputpath") + ".output"
        }
        sc.write(simpleKMSDevelopers, cryptoMode = CryptoMode.parse(
          conf.get("spark.bigdl.kms.datasource1.outputEncryptMode")))
          .mode("overwrite")
          .option("header", true)
          .csv(simpleKMSOutputPath, conf.get("spark.bigdl.kms.datasource1.data"))

        var ehsmOutputPath: String = ""
        if (conf.contains("spark.bigdl.kms.datasource2.outputpath")){
          ehsmOutputPath = conf.get("spark.bigdl.kms.datasource2.outputpath")
        } else {
          ehsmOutputPath = conf.get("spark.bigdl.kms.datasource2.inputpath") + ".output"
        }
        sc.write(ehsmDevelopers, cryptoMode = CryptoMode.parse(
          conf.get("spark.bigdl.kms.datasource2.outputEncryptMode")))
          .mode("overwrite")
          .option("header", true)
          .csv(ehsmOutputPath, conf.get("spark.bigdl.kms.datasource2.data"))
      }
    }
  }

}
