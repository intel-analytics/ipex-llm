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
package com.intel.analytics.bigdl.ppml.crypto.dataframe

import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.PLAIN_TEXT
import org.apache.spark.SparkConf

class EncryptedJsonSpec extends DataFrameHelper {
  override val repeatedNum = 2000

  val ppmlArgs = Map(
    "spark.bigdl.kms.simple.id" -> appid,
    "spark.bigdl.kms.simple.key" -> appkey,
    "spark.bigdl.kms.key.primary" -> primaryKeyPath,
    "spark.bigdl.kms.key.data" -> dataKeyPath
  )
  val conf = new SparkConf().setMaster("local[4]")
  conf.set("spark.hadoop.io.compression.codecs",
    "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec")
  val sc = PPMLContext.initPPMLContext(conf, "SimpleQuery", ppmlArgs)

  val sparkSession = sc.getSparkSession()
  import sparkSession.implicits._

  "json" should "work" in {
    val plainJsonPath = dir + "/plain-json"
    val df = sc.read(cryptoMode = PLAIN_TEXT)
      .option("header", "true").csv(plainFileName)
    df.write
      .option("compression", "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec")
      .json(plainJsonPath)
    val jsonDf = sparkSession.read.json(plainJsonPath)
    jsonDf.count()

  }

  "csv" should "work" in {
    val plainJsonPath = dir + "/plain-csv"
    val df = sc.read(cryptoMode = PLAIN_TEXT)
      .option("header", "true").csv(plainFileName)
    df.write
      .option("compression", "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec")
//      .option("compression", "deflate")
      .csv(plainJsonPath)
    val jsonDf = sparkSession.read.csv(plainJsonPath)
    jsonDf.count()

  }

}