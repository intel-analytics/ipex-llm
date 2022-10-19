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
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, BigDLEncrypt, CryptoCodec, ENCRYPT}
import com.intel.analytics.bigdl.ppml.kms.SimpleKeyManagementService
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

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

  def generateCsvData(): (String, String, String, String) = {
    Random.setSeed(1)
    val fileName = dir + "/people.csv"
    val encryptFileName = dir + "/en_people.csv" + CryptoCodec.getDefaultExtension()
    val fw = new FileWriter(fileName)
    val data = new StringBuilder()
    data.append(header)
    (0 until repeatedNum).foreach {i =>
      data.append(s"gdni,$i,Engineer\npglyal,$i,Engineer\nyvomq,$i,Developer\n")
    }
    fw.append(data)
    fw.close()

    simpleKms.retrievePrimaryKey(primaryKeyPath)
    simpleKms.retrieveDataKey(primaryKeyPath, dataKeyPath)
    logger.info("write primaryKey to " + primaryKeyPath)
    val crypto = new BigDLEncrypt()
    val dataKeyPlaintext = simpleKms.retrieveDataKeyPlainText(primaryKeyPath, dataKeyPath)
    crypto.init(AES_CBC_PKCS5PADDING, ENCRYPT, dataKeyPlaintext)
    crypto.doFinal(fileName, encryptFileName)
    (fileName, encryptFileName, data.toString(), dataKeyPlaintext)
  }

}
