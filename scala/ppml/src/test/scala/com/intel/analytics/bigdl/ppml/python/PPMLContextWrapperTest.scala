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

package com.intel.analytics.bigdl.ppml.python

import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, BigDLEncrypt, ENCRYPT}
import com.intel.analytics.bigdl.ppml.kms.SimpleKeyManagementService
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import scala.collection.JavaConverters._

import java.io.{File, PrintWriter}

class PPMLContextWrapperTest extends FunSuite with BeforeAndAfterAll{
  val ppmlContextWrapper: PPMLContextWrapper[Float] = PPMLContextWrapper.ofFloat
  val kms: SimpleKeyManagementService = SimpleKeyManagementService.apply()
  val appName = "test"
  val ppmlArgs: Map[String, String] = Map(
    "kms_type" -> "SimpleKeyManagementService",
    "simple_app_id" -> kms._appId,
    "simple_app_key" -> kms._appKey,
    "primary_key_path" -> (this.getClass.getClassLoader.getResource("").getPath + "primaryKey"),
    "data_key_path" -> (this.getClass.getClassLoader.getResource("").getPath + "dataKey")
  )
  var sc: PPMLContext = null

  override def beforeAll(): Unit = {
    // generate a tmp csv file
    val csvFile = this.getClass.getClassLoader.getResource("").getPath + "people.csv"
    val csvWriter = new PrintWriter(new File(csvFile))
    csvWriter.println("name,age,job")
    csvWriter.println("jack,18,Developer")
    csvWriter.println("alex,20,Researcher")
    csvWriter.println("xuoui,25,Developer")
    csvWriter.println("hlsgu,29,Researcher")
    csvWriter.println("xvehlbm,45,Developer")
    csvWriter.println("ehhxoni,23,Developer")
    csvWriter.println("capom,60,Developer")
    csvWriter.println("pjt,24,Developer")
    csvWriter.close()

    // generate a primaryKey and dataKey
    val primaryKeyPath = this.getClass.getClassLoader.getResource("").getPath + "primaryKey"
    val dataKeyPath = this.getClass.getClassLoader.getResource("").getPath + "dataKey"
    kms.retrievePrimaryKey(primaryKeyPath)
    kms.retrieveDataKey(primaryKeyPath, dataKeyPath)

    // generate encrypted file
    val encryptedFilePath = this.getClass.getClassLoader.getResource("").getPath + "encrypted/people.csv"
    val dataKeyPlaintext = kms.retrieveDataKeyPlainText(primaryKeyPath, dataKeyPath)
    val encrypt = new BigDLEncrypt()
    encrypt.init(AES_CBC_PKCS5PADDING, ENCRYPT, dataKeyPlaintext)
    encrypt.doFinal(csvFile, encryptedFilePath)

    // init PPMLContext
    sc = ppmlContextWrapper.createPPMLContext(appName, ppmlArgs.asJava)
  }

  override def afterAll(): Unit = {
    val csvFile = new File(this.getClass.getClassLoader.getResource("").getPath + "people.csv")
    val primaryKey = new File(this.getClass.getClassLoader.getResource("").getPath + "primaryKey")
    val dataKey = new File(this.getClass.getClassLoader.getResource("").getPath + "dataKey")
    val encryptPath = new File(this.getClass.getClassLoader.getResource("").getPath + "encrypted")
    val writeOutput = new File(this.getClass.getClassLoader.getResource("").getPath + "output")
    if (csvFile.isFile) {
      csvFile.delete()
    }
    if (dataKey.isFile) {
      dataKey.delete()
    }
    if (primaryKey.isFile) {
      primaryKey.delete()
    }
    if (encryptPath.isDirectory) {
      deleteDir(encryptPath)
    }
    if (writeOutput.isDirectory) {
      deleteDir(writeOutput)
    }
  }

  def deleteDir(dir: File): Unit = {
    val files = dir.listFiles()
    files.foreach(file => {
      if (file.isDirectory) {
        deleteDir(file)
      } else {
        file.delete()
      }
    })
    dir.delete()
  }

  def read(cryptoMode: String, path: String): Unit = {
    val encryptedDataFrameReader = ppmlContextWrapper.read(sc, cryptoMode)
    ppmlContextWrapper.option(encryptedDataFrameReader, "header", "true")
    val df = ppmlContextWrapper.csv(encryptedDataFrameReader, path)

    Log4Error.invalidOperationError(df.count() == 8,
        "record count should be 8")
  }

  def write(df: DataFrame, encryptMode: String): Unit = {
    var outputDir = "output/"
    if (encryptMode == "AES/CBC/PKCS5Padding") {
      outputDir = outputDir + "encrypt"
    } else {
      outputDir = outputDir + "plain"
    }
    val path = this.getClass.getClassLoader.getResource("").getPath + outputDir

    val encryptedDataFrameWriter = ppmlContextWrapper.write(sc, df, encryptMode)
    ppmlContextWrapper.mode(encryptedDataFrameWriter, "overwrite")
    ppmlContextWrapper.option(encryptedDataFrameWriter, "header", true)
    ppmlContextWrapper.csv(encryptedDataFrameWriter, path)
  }

  test("init PPMLContext with app name") {
    ppmlContextWrapper.createPPMLContext(appName)
  }

  test("init PPMLContext with app name & args") {
    ppmlContextWrapper.createPPMLContext(appName, ppmlArgs.asJava)
  }

  test("read plain text csv file") {
    val cryptoMode = "plain_text"
    val path = this.getClass.getClassLoader.getResource("people.csv").getPath

    read(cryptoMode, path)
  }

  test("read encrypted csv file") {
    val cryptoMode = "AES/CBC/PKCS5Padding"
    val path = this.getClass.getClassLoader.getResource("") + "encrypted/people.csv"

    read(cryptoMode, path)
  }

  test(" write plain text csv file") {
    val spark: SparkSession = SparkSession.builder()
      .master("local[1]").appName("testData")
      .getOrCreate()

    val data = Seq(("Java", "20000"), ("Python", "100000"), ("Scala", "3000"))

    val df = spark.createDataFrame(data).toDF("language", "user")

    write(df, "plain_text")
  }

  test(" write encrypted csv file") {
    val spark: SparkSession = SparkSession.builder()
      .master("local[1]").appName("testData")
      .getOrCreate()

    val data = Seq(("Java", "20000"), ("Python", "100000"), ("Scala", "3000"))

    val df = spark.createDataFrame(data).toDF("language", "user")

    write(df, "AES/CBC/PKCS5Padding")
  }

}
