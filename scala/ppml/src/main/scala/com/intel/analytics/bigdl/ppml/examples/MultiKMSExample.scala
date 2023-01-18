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
import com.intel.analytics.bigdl.ppml.utils.{EncryptIOArguments, Supportive, DataSource, DataSink}
import com.intel.analytics.bigdl.ppml.crypto.{CryptoMode, EncryptRuntimeException, PLAIN_TEXT, AES_CBC_PKCS5PADDING}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.slf4j.LoggerFactory

object MultiDataSourceExample extends Supportive {
  def main(args: Array[String]): Unit = {
    // get spark session and make ppml context
    val sparkSession = SparkSession.builder().getOrCreate
    val conf = sparkSession.sparkContext.getConf
    val sc = PPMLContext.initPPMLMultiKMSContext(sparkSession)

    timing("processing") {
      // load csv file to data frame with ppmlcontext.
      val amyDf = timing("1/3 read amy data source") {
        val amyDataSource = DataSource("simpleKMS",
                                       "./amyPrimaryKeyPath",
                                       "./amyDataKeyPath",
                                       AES_CBC_PKCS5PADDING)
        sc.read(amyDataSource)
          .option("header", "true")
          .csv("./amyDataSourcePath")
      }

      val bobDf = timing("1/3 read bob data source") {
        val bobDataSource = DataSource("ehsmKMS",
                                       "./bobPrimaryKeyPath",
                                       "./bobDataKeyPath",
                                       AES_CBC_PKCS5PADDING)
        sc.read(bobDataSource)
          .option("header", "true")
          .csv("./bobDataSourcePath")
      }


      val amyDevelopers = timing("2/3 doSQLOperations") {
        // Select only the "name" column
        amyDf.select("name").count()

        // Select everybody, but increment the age by 1
        amyDf.select(amyDf("name"), amyDf("age") + 1).show()

        // Select Developer and records count
        val amyDevelopers = amyDf.filter(
          amyDf("job") === "Developer" and amyDf("age").between(20, 40))
          .toDF()
        amyDevelopers.count()

        amyDevelopers
      }

      val bobDevelopers = timing("2/3 datasource2 do SQL") {
        bobDf.select("name").count
        bobDf.select(bobDf("name"), bobDf("age") ).show()
        val bobDevelopers = bobDf.filter(bobDf("job") === "Developer" and bobDf("age")
          .between(20, 40)).toDF()
        bobDevelopers.count()
        bobDevelopers
      }


      timing("3/3 encryptAndSaveOutputs") {
        // save data frame using spark kms context
        // write encrypted data
        val dataSink = DataSink("ehsmKMS",
                                "./bobPrimaryKeyPath",
                                "./bobDataKeyPath",
                                AES_CBC_PKCS5PADDING)
        sc.write(bobDevelopers, dataSink)
          .mode("overwrite")
          .option("header", true)
          .csv("./outputPath")
      }
    }
  }

}
