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
      "spark.bigdl.enableMultiKms" -> "true",
      "spark.bigdl.kms.amyKms.type" -> "EHSMKeyManagementService",
      "spark.bigdl.kms.amyKms.ip" -> args(0),
      "spark.bigdl.kms.amyKms.port" -> args(1),
      "spark.bigdl.kms.amyKms.appId" -> args(2),
      "spark.bigdl.kms.amyKms.apiKey" -> args(3),
      "spark.bigdl.kms.bobKms.type" -> "EHSMKeyManagementService",
      "spark.bigdl.kms.bobKms.ip" -> args(4),
      "spark.bigdl.kms.bobKms.port" -> args(5),
      "spark.bigdl.kms.bobKms.appId" -> args(6),
      "spark.bigdl.kms.bobKms.apiKey" -> args(7)
    )
    val sc = PPMLContext.initPPMLContext("MultiPartySparkEncryptExample", ppmlArgs)

    timing("loading") {
      // load csv file to data frame with ppmlcontext.
      val amyDf = timing("1/4 read Amy's plaintext data source into data frame") {
        // read encrypted data
        sc.read(PLAIN_TEXT, // crypto mode
                "amyKms", // name of kms which data key is retrieved from
                "./amy_encrypted_primary_key", // primary key file path
                "./amy_encrypted_data_key") // data key file path
          .option("header", "true")
          .csv("./amyDataSource.csv") // input file path
      }

      val bobDf = timing("2/4 read Bob's plaintext data source") {
        sc.read(PLAIN_TEXT, "bobKms",
                "./bob_encrypted_primary_key", "./bob_encrypted_data_key")
          .option("header", "true")
          .csv("./bobDataSource.csv")
      }

      timing("3/4 encrypt and save amy data") {
        // save data frame using spark kms context
        // write encrypted data
        sc.write(amyDf, // target data frame
                 AES_CBC_PKCS5PADDING, // encrypt mode
                 "amyKms", // name of kms which data key is retrieved from
                 "./amy_encrypted_primary_key", // primary key file path
                 "./amy_encrypted_data_key") // data key file path
          .mode("overwrite")
          .option("header", true)
          .csv("./encrypted_amy_data")
      }

      timing("4/4 encrypt and save Bob encrypted data") {
        sc.write(bobDf, AES_CBC_PKCS5PADDING, "bobKms",
                 "./bob_encrypted_primary_key", "./bob_encrypted_data_key")
          .mode("overwrite").option("header", true).csv("./encrypted_bob_data")
      }
    }
  }
}

