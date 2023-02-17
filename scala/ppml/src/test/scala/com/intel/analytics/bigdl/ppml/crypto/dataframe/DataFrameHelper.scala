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

import com.intel.analytics.bigdl.dllib.common.zooUtils
import com.intel.analytics.bigdl.ppml.BigDLSpecHelper
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, BigDLEncrypt, CryptoCodec, ENCRYPT, DECRYPT, PLAIN_TEXT}
import com.intel.analytics.bigdl.ppml.kms.SimpleKeyManagementService
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.ppml.PPMLContext
import org.apache.spark.{SparkConf, SparkContext}

import java.io.FileWriter
import scala.util.Random

class DataFrameHelper extends BigDLSpecHelper {
  val repeatedNum = 100000
  val totalNum = repeatedNum * 3
  val header = "name,age,job\n"
  val (appid, apikey) = generateKeys()
  val simpleKms = SimpleKeyManagementService(appid, apikey)
  val dir = createTmpDir("rwx------")

  val primaryKeyPath = dir + s"/primary.key"
  val dataKeyPath = dir + s"/data.key"

  def generateKeys(): (String, String) = {
    val appid: String = "123456789012"
    val apikey: String = "210987654321"
    (appid, apikey)
  }

  def generateCsvData(): (String, String, String) = {
    Random.setSeed(1)
    val fileName = dir + "/people.csv"
    val encryptFileName = dir + "/people.encrypted"
    val fw = new FileWriter(fileName)
    val data = new StringBuilder()
    data.append(header)
    (0 until repeatedNum).foreach {i =>
      data.append(s"gdni,$i,Engineer\npglyal,$i,Engineer\nyvomq,$i,Developer\n")
    }
    fw.append(data)
    fw.close()

    simpleKms.retrievePrimaryKey(primaryKeyPath)
    logger.info("write primaryKey to " + primaryKeyPath)

    val ppmlArgs = Map(
      "spark.bigdl.primaryKey.defaultKey.kms.type" -> "SimpleKeyManagementService",
      "spark.bigdl.primaryKey.defaultKey.kms.appId" -> appid,
      "spark.bigdl.primaryKey.defaultKey.kms.apiKey" -> apikey,
      "spark.bigdl.primaryKey.defaultKey.material" -> primaryKeyPath
    )
    val sparkConf = new SparkConf().setMaster("local[4]")
    val sc = PPMLContext.initPPMLContext(sparkConf, "SimpleQuery", ppmlArgs)
    
    logger.info("read source csv in to PPMLContext sc")
    val df = sc.read(PLAIN_TEXT).csv(fileName)
    logger.info("encrypt and write csv in to PPMLContext sc")
    sc.write(df, AES_CBC_PKCS5PADDING, "defaultKey").mode("overwrite").csv(encryptFileName)
    (fileName, encryptFileName, data.toString())
  }

}
