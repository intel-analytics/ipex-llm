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

object MultiPartySparkQueryExample extends Supportive {
  def main(args: Array[String]): Unit = {
    val sparkSession: SparkSession = SparkSession.builder().getOrCreate()
    val (amyEncryptedDataFileInputPath,
         bobEncryptedDataFileInputPath) = (args(0), args(1))

    var (amyEncryptedDataFileOutputPath,
    bobEncryptedDataFileOutputPath) = ("", "")
    if (args.length == 4) {
      amyEncryptedDataFileOutputPath = args(2)
      bobEncryptedDataFileOutputPath = args(3)
    }

    val sc = PPMLContext.initPPMLContext(sparkSession)

    timing("processing") {
      // load csv file to data frame with ppmlcontext.
      val amyDf = timing("1/8 read Amy's data source into data frame") {
        // read encrypted data
        sc.read(cryptoMode = AES_CBC_PKCS5PADDING,
                primaryKeyName = "AmyPK")
          .option("header", "true")
          .csv(amyEncryptedDataFileInputPath)
      }

      val bobDf = timing("2/8 read Bob's data source") {
        sc.read(AES_CBC_PKCS5PADDING, "BobPK")
          .option("header", "true")
          .csv(bobEncryptedDataFileInputPath)
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
        sc.write(dataFrame = unionDf,
                 cryptoMode = AES_CBC_PKCS5PADDING,
                 primaryKeyName = "AmyPK")
          .mode("overwrite")
          .option("header", true)
          .csv(amyEncryptedDataFileOutputPath + "./union_output")
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
        sc.write(joinDf, AES_CBC_PKCS5PADDING, "BobPK")
          .mode("overwrite").option("header", true)
          .csv(bobEncryptedDataFileOutputPath + "./join_output")
      }
    }
  }
}

