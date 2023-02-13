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
import com.intel.analytics.bigdl.ppml.crypto.{CryptoMode, AES_CBC_PKCS5PADDING, PLAIN_TEXT}
import org.apache.spark.sql.SparkSession

object MultiPartySparkEncryptExample extends Supportive {
  def main(args: Array[String]): Unit = {

    val ppmlArgs = Map(
      "spark.bigdl.primaryKey.AmyPK.plainText" -> args(0),
      "spark.bigdl.primaryKey.BobPK.kms.type" -> "EHSMKeyManagementService",
      // path of ehsm encrypted_primary_key file
      "spark.bigdl.primaryKey.BobPK.material" -> args(1),
      "spark.bigdl.primaryKey.BobPK.kms.ip" -> args(2),
      "spark.bigdl.primaryKey.BobPK.kms.port" -> args(3),
      "spark.bigdl.primaryKey.BobPK.kms.appId" -> args(4),
      "spark.bigdl.primaryKey.BobPK.kms.apiKey" -> args(5)
    )
    val (amy_data_file_input_path, bob_data_file_input_path) = (args(6), args(7))
    val sc = PPMLContext.initPPMLContext("MultiPartySparkEncryptExample", ppmlArgs)

    timing("loading") {
      // load csv file to data frame with ppmlcontext.
      val amyDf = timing("1/4 read Amy's plaintext data source into data frame") {
        // read encrypted data
        sc.read(PLAIN_TEXT, // crypto mode
                "AmyPK") // primary key name
          .option("header", "true")
          .csv(amy_data_file_input_path)
      }

      val bobDf = timing("2/4 read Bob's plaintext data source") {
        sc.read(PLAIN_TEXT, "BobPK")
          .option("header", "true")
          .csv(bob_data_file_input_path)
      }

      timing("3/4 encrypt and save Amy data") {
        // save data frame using spark kms context
        // write encrypted data
        sc.write(amyDf, // target data frame
                 AES_CBC_PKCS5PADDING, // encrypt mode
                 "AmyPK") // primary key name
          .mode("overwrite")
          .option("header", true)
          .csv("./encrypted_amy_data")
      }

      timing("4/4 encrypt and save Bob encrypted data") {
        sc.write(bobDf, AES_CBC_PKCS5PADDING, "BobPK")
          .mode("overwrite").option("header", true).csv("./encrypted_bob_data")
      }
    }
  }
}

