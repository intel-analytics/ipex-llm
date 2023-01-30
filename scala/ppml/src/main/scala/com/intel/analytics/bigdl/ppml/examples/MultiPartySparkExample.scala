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
import com.intel.analytics.bigdl.ppml.utils.Supportive
import com.intel.analytics.bigdl.ppml.crypto.{CryptoMode, AES_CBC_PKCS5PADDING}
import org.apache.spark.sql.SparkSession

object MultiPartySparkExample extends Supportive {
  def main(args: Array[String]): Unit = {

    val sparkSession = SparkSession.builder().getOrCreate
    // load spark configurations into ppml context
    val conf = sparkSession.sparkContext.getConf
    val sc = PPMLContext.initPPMLContext(conf, "MultiPartySparkExample")

    timing("processing") {
      // load csv file to data frame with ppmlcontext.
      val amyDf = timing("1/8 read Amy's data source into data frame") {
        // read encrypted data
        sc.read(AES_CBC_PKCS5PADDING, // crypto mode
                "amyKms", // name of kms which data key is retrieved from
                "./amy_encrypted_primary_key", // primary key file path
                "./amy_encrypted_data_key") // data key file path
          .option("header", "true")
          .csv("./amyDataSource.csv") // input file path
      }

      val bobDf = timing("2/8 read Bob's data source") {
        sc.read(AES_CBC_PKCS5PADDING, "bobKms",
                "./bob_encrypted_primary_key", "./bob_encrypted_data_key")
          .option("header", "true")
          .csv("./bobDataSource.csv")
      }

      val amyDevelopers = timing("3/8 do SQL operations on Amy data frame") {
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

      val bobDevelopers = timing("4/8 do SQL operations on Bob data frame") {
        bobDf.select(bobDf("name"), bobDf("age") ).show()
        val bobDevelopers = bobDf.filter(
          bobDf("job") === "Developer" and bobDf("age").between(20, 40))
          .toDF()
        bobDevelopers
      }

      val unionDf = timing("5/8 union Amy developers and Bob developers") {
        amyDevelopers.union(bobDevelopers)
      }

      timing("6/8 encrypt and save union outputs") {
        // save data frame using spark kms context
        // write encrypted data
        sc.write(unionDf, // target data frame
                 AES_CBC_PKCS5PADDING, // encrypt mode
                 "sharedKms", // name of kms which data key is retrieved from
                 "./shared_encrypted_primary_key", // primary key file path
                 "./shared_encrypted_data_key") // data key file path
          .mode("overwrite")
          .option("header", true)
          .csv("./union_output")
      }

      val joinDf = timing("7/8 join Amy developers and Bob developers on age") {
        val (amyColNames, bobColNames) = (Seq("amyDeveloperName",
                                              "amyDeveloperAge", "amyDeveloperJob"),
                                         Seq("bobDeveloperName",
                                             "bobDeveloperAge", "bobDeveloperJob"))
        val (amyDevRenamedCol, bobDevRenamedCol) = (amyDevelopers.toDF(amyColNames: _*),
                                                    bobDevelopers.toDF(bobColNames: _*))
        amyDevRenamedCol.join(bobDevRenamedCol,
          amyDevRenamedCol("amyDeveloperAge") === bobDevRenamedCol("bobDeveloperAge"),
          "inner")
      }

      timing("6/8 encrypt and save join outputs") {
        sc.write(joinDf, AES_CBC_PKCS5PADDING, "sharedKms",
                 "./shared_encrypted_primary_key", "./shared_encrypted_data_key")
          .mode("overwrite").option("header", true).csv("./join_output")
      }
    }
  }
}

