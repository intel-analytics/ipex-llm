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
  val (plainFileName, encryptFileName, data, dataKeyPlaintext) = generateCsvData()

  val ppmlArgs = Map(
    "spark.bigdl.kms.simple.id" -> appid,
    "spark.bigdl.kms.simple.key" -> apikey,
    "spark.bigdl.kms.key.primary" -> primaryKeyPath,
    "spark.bigdl.kms.key.data" -> dataKeyPath
  )
  val conf = new SparkConf().setMaster("local[4]")
  conf.set("spark.hadoop.io.compression.codecs",
    "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec")
  val sc = PPMLContext.initPPMLContext(conf, "SimpleQuery", ppmlArgs)

  val sparkSession = sc.getSparkSession()

  "spark session read/write json" should "work" in {
    val encryptJsonPath = dir + "/en-json"
    val df = sc.read(cryptoMode = PLAIN_TEXT)
      .option("header", "true").csv(plainFileName)
    df.write
      .option("compression", "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec")
      .json(encryptJsonPath)
    val jsonDf = sparkSession.read.json(encryptJsonPath)
    jsonDf.count()
    val d = "name,age,job\n" +
      df.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d + "\n" should be (data)
  }

  "PPMLContext read/write json" should "work" in {
    val encryptJsonPath = dir + "/en-json"
    val df = sc.read(cryptoMode = PLAIN_TEXT)
      .option("header", "true").csv(plainFileName)
    df.write
      .option("compression", "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec")
      .json(encryptJsonPath)
    val jsonDf = sparkSession.read.json(encryptJsonPath)
    jsonDf.count()
    val d = "name,age,job\n" +
      df.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d + "\n" should be (data)
  }


  "ppml context json read/write different size" should "work" in {
    val filteredPath = dir + "/filtered-json"
    val df = sc.read(cryptoMode = PLAIN_TEXT)
      .option("header", "true").csv(plainFileName)
    df.count() should be (repeatedNum * 3)
    (1 to 10).foreach{ i =>
      val step = 1000
      val filtered = df.filter(_.getString(1).toInt < i * step)
      val filteredData = df.collect()
      sc.write(filtered, AES_CBC_PKCS5PADDING).mode("overwrite").json(filteredPath)
      val readed = sc.read(AES_CBC_PKCS5PADDING).json(filteredPath)
      readed.count() should be (i * step * 3)
      readed.collect().zip(filteredData).foreach{v =>
        v._1.getAs[String]("age") should be (v._2.getAs[String]("age"))
        v._1.getAs[String]("job") should be (v._2.getAs[String]("job"))
        v._1.getAs[String]("name") should be (v._2.getAs[String]("name"))
      }
    }
  }
}
