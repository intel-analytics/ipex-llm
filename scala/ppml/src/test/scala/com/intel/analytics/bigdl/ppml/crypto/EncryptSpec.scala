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

package com.intel.analytics.bigdl.ppml.crypto

import com.intel.analytics.bigdl.dllib.common.zooUtils
import com.intel.analytics.bigdl.dllib.utils.File
import com.intel.analytics.bigdl.ppml.kms.SimpleKeyManagementService
import org.apache.hadoop.fs.Path
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import java.io.{BufferedReader, BufferedWriter, FileReader, FileWriter}
import java.nio.file.{Files, Paths, StandardOpenOption}
import scala.util.Random

class EncryptSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val (appid, appkey) = generateKeys()
  val simpleKms = SimpleKeyManagementService(appid, appkey)
  val dir = zooUtils.createTmpDir("PPMLUT", "rwx------").toFile()
  val primaryKeyPath = dir + "/primary.key"
  val dataKeyPath = dir + "/data.key"
  simpleKms.retrievePrimaryKey(primaryKeyPath)
  simpleKms.retrieveDataKey(primaryKeyPath, dataKeyPath)
  var dataKeyPlaintext: String = null
  val (plainFileName, encryptFileName, data) = generateCsvData()
  val fs = File.getFileSystem(plainFileName)

  def generateKeys(): (String, String) = {
    val appid: String = (1 to 12).map(x => Random.nextInt(10)).mkString
    val appkey: String = (1 to 12).map(x => Random.nextInt(10)).mkString
    (appid, appkey)
  }
  def generateCsvData(): (String, String, String) = {
    val fileName = dir + "/people.csv"
    val encryptFileName = dir + "/en_people.csv"
    val fw = new FileWriter(fileName)
    val data = new StringBuilder()
    data.append(s"name,age,job\n")
    data.append(s"yvomq,59,Developer\ngdni,40,Engineer\npglyal,33,Engineer")
    fw.append(data)
    fw.close()

    val crypto = new BigDLEncrypt()
    dataKeyPlaintext = simpleKms.retrieveDataKeyPlainText(primaryKeyPath, dataKeyPath)
    crypto.init(AES_CBC_PKCS5PADDING, ENCRYPT, dataKeyPlaintext)
    Files.write(Paths.get(encryptFileName), crypto.genHeader())
    val encryptedBytes = crypto.doFinal(data.toString().getBytes)
    Files.write(Paths.get(encryptFileName), encryptedBytes._1, StandardOpenOption.APPEND)
    Files.write(Paths.get(encryptFileName), encryptedBytes._2, StandardOpenOption.APPEND)
    (fileName, encryptFileName, data.toString())
  }

  "encrypt stream" should "work" in {
    val encrypt = new BigDLEncrypt()
    encrypt.init(AES_CBC_PKCS5PADDING, ENCRYPT, dataKeyPlaintext)
    val bis = fs.open(new Path(plainFileName))
    val outs = fs.create(new Path(dir + "/en_o.csv"))
    encrypt.doFinal(bis, outs)
    bis.close()
    outs.flush()
    outs.close()
    Thread.sleep(1000)

    val decrypt = new BigDLEncrypt()
    decrypt.init(AES_CBC_PKCS5PADDING, DECRYPT, dataKeyPlaintext)
    val bis2 = fs.open(new Path(dir + "/en_o.csv"))
    val outs2 = fs.create(new Path(dir + "/de_o.csv"))
    decrypt.doFinal(bis2, outs2)
    outs2.close()
    outs2.flush()
    bis2.close()
    val originFile = Files.readAllBytes(Paths.get(plainFileName))
    val deFile = Files.readAllBytes(Paths.get(dir.toString, "/de_o.csv"))
    originFile.sameElements(deFile) should be (true)
  }
}
