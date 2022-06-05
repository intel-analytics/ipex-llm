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
import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, BigDLEncrypt, CryptoMode, DECRYPT, ENCRYPT, PLAIN_TEXT}
import com.intel.analytics.bigdl.ppml.kms.SimpleKeyManagementService
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import java.io.FileWriter
import java.nio.file.{Files, Paths, StandardOpenOption}
import scala.util.Random

class EncryptDataFrameSpec extends FlatSpec with Matchers with BeforeAndAfter{
  val (appid, appkey) = generateKeys()
  val simpleKms = SimpleKeyManagementService(appid, appkey)
  val dir = zooUtils.createTmpDir("PPMLUT", "rwx------").toFile()

  val primaryKeyPath = dir + "/primary.key"
  val dataKeyPath = dir + "/data.key"
  simpleKms.retrievePrimaryKey(primaryKeyPath)
  simpleKms.retrieveDataKey(primaryKeyPath, dataKeyPath)
  val (plainFileName, encryptFileName, data) = generateCsvData()
  val repeatedNum = 500000
  val totalNum = repeatedNum * 3 + 1

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
    (0 until 500000).foreach {i =>
//    (0 to 500).foreach {i =>
      data.append(s"yvomq,$i,Developer\ngdni,$i,Engineer\npglyal,$i,Engineer\n")
    }
//    data.append(s"yvomq,59,Developer\ngdni,40,Engineer\npglyal,33,Engineer")
    fw.append(data)
    fw.close()

    val crypto = new BigDLEncrypt()
    val dataKeyPlaintext = simpleKms.retrieveDataKeyPlainText(primaryKeyPath, dataKeyPath)
    crypto.init(AES_CBC_PKCS5PADDING, ENCRYPT, dataKeyPlaintext)
    crypto.doFinal(fileName, encryptFileName)
//    Files.write(Paths.get(encryptFileName), crypto.genHeader())
//    val encryptedBytes = crypto.doFinal(data.toString().getBytes)
//    Files.write(Paths.get(encryptFileName), encryptedBytes._1, StandardOpenOption.APPEND)
//    Files.write(Paths.get(encryptFileName), encryptedBytes._2, StandardOpenOption.APPEND)
    crypto.init(AES_CBC_PKCS5PADDING, DECRYPT, dataKeyPlaintext)
    crypto.doFinal(encryptFileName, "/tmp/123.csv")
    (fileName, encryptFileName, data.toString())
  }
  val ppmlArgs = Map(
      "spark.bigdl.kms.simple.id" -> appid,
      "spark.bigdl.kms.simple.key" -> appkey,
      "spark.bigdl.kms.key.primary" -> primaryKeyPath,
      "spark.bigdl.kms.key.data" -> dataKeyPath
  )
  val sparkConf = new SparkConf().setMaster("local[4]")
  val sc = PPMLContext.initPPMLContext(sparkConf, "SimpleQuery", ppmlArgs)

  "textfile read from plaint text file" should "work" in {
    val file = sc.textFile(plainFileName).collect()
    file.mkString("\n") should be (data)
    val file2 = sc.textFile(encryptFileName, cryptoMode = AES_CBC_PKCS5PADDING).collect()
    file2.mkString("\n") should be (data)
  }

  "sparkSession.read" should "work" in {
    val sparkSession: SparkSession = SparkSession.builder().getOrCreate()
    val df = sparkSession.read.csv(plainFileName)
    val d = df.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d should be (data)
    val df2 = sparkSession.read.option("header", "true").csv(plainFileName)
    val d2 = df2.schema.map(_.name).mkString(",") + "\n" +
      df2.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d2 should be (data)
  }

  "read from plain csv with header" should "work" in {
    val df = sc.read(cryptoMode = PLAIN_TEXT)
      .option("header", "true").csv(plainFileName)
    val d = df.schema.map(_.name).mkString(",") + "\n" +
      df.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d should be (data)
  }

  "read from encrypted csv with header" should "work" in {
    val df = sc.read(cryptoMode = AES_CBC_PKCS5PADDING)
      .option("header", "true").csv(encryptFileName)
    val d = df.schema.map(_.name).mkString(",") + "\n" +
      df.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d should be (data)
  }

  "read from plain csv without header" should "work" in {
    val df = sc.read(cryptoMode = PLAIN_TEXT).csv(plainFileName)
    val d = df.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d should be (data)
  }

  "read from encrypted csv without header" should "work" in {
    val df = sc.read(cryptoMode = AES_CBC_PKCS5PADDING).csv(encryptFileName)
    val d = df.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d should be (data)
  }

  "save df" should "work" in {
    val enWriteCsvPath = dir + "/en_write_csv"
    val writeCsvPath = dir + "/write_csv"
    val df = sc.read(cryptoMode = AES_CBC_PKCS5PADDING).csv(encryptFileName)
//    val df = sc.read(cryptoMode = PLAIN_TEXT).csv(plainFileName)
    df.count() should be (totalNum)
    sc.write(df, cryptoMode = AES_CBC_PKCS5PADDING).csv(enWriteCsvPath)
    sc.write(df, cryptoMode = PLAIN_TEXT).csv(writeCsvPath)

    val readEn = sc.read(cryptoMode = AES_CBC_PKCS5PADDING).csv(enWriteCsvPath)
    val readEnCollect = readEn.collect().map(v =>
      s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    readEnCollect + "\n" should be (data)

    val readPlain = sc.read(cryptoMode = PLAIN_TEXT).csv(writeCsvPath)
    val readPlainCollect = readPlain.collect().map(v =>
      s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    readPlainCollect + "\n" should be (data)
  }
}

