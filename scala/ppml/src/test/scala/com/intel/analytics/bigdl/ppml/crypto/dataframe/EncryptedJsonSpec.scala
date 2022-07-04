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
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, Crypto, DECRYPT, PLAIN_TEXT}
import org.apache.spark.SparkConf

import java.io.File

class EncryptedJsonSpec extends DataFrameHelper {
  override val repeatedNum = 160

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

  "json read/write" should "work" in {
    val plainJsonPath = dir + "/plain-csv"
    val df = sc.read(cryptoMode = PLAIN_TEXT)
      .option("header", "true").csv(plainFileName)
    df.write
      .option("compression", "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec")
      .json(plainJsonPath)
    val jsonDf = sparkSession.read.json(plainJsonPath)
    jsonDf.count()
    val d = "name,age,job\n" +
      df.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d + "\n" should be (data)
  }

  "csv read/write" should "work" in {
    val plainJsonPath = dir + "/plain-csv"
//    val plainJsonPath = "/tmp/3/e1"
    val df = sc.read(cryptoMode = PLAIN_TEXT)
      .option("header", "true").csv(plainFileName)
    df.write
      .option("compression", "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec")
      //      .option("compression", "deflate")
      .csv(plainJsonPath)
    val jsonDf = sc.read(AES_CBC_PKCS5PADDING).csv(plainJsonPath)
    jsonDf.count() should be (repeatedNum * 3)
    val d = "name,age,job\n" +
      jsonDf.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d + "\n" should be (data)
  }

}