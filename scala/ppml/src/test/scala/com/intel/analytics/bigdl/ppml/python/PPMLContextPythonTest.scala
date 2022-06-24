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

class PPMLContextPythonTest extends FunSuite with BeforeAndAfterAll{
  val ppmlContextPython: PPMLContextPython[Float] = PPMLContextPython.ofFloat
  val kms: SimpleKeyManagementService = SimpleKeyManagementService.apply()
  val appName = "test"
  val dir: String = this.getClass.getClassLoader.getResource("").getPath
  val ppmlArgs: Map[String, String] = Map(
    "kms_type" -> "SimpleKeyManagementService",
    "simple_app_id" -> kms._appId,
    "simple_app_key" -> kms._appKey,
    "primary_key_path" -> (dir + "primaryKey"),
    "data_key_path" -> (dir + "dataKey")
  )
  var sc: PPMLContext = null
  var df: DataFrame = null
  var dataContent: String = null

  override def beforeAll(): Unit = {
    // generate a tmp csv file
    val csvFile = dir + "people.csv"
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
    val primaryKeyPath = dir + "primaryKey"
    val dataKeyPath = dir + "dataKey"
    kms.retrievePrimaryKey(primaryKeyPath)
    kms.retrieveDataKey(primaryKeyPath, dataKeyPath)

    // generate encrypted file
    val encryptedFilePath = dir + "encrypted/people.csv"
    val dataKeyPlaintext = kms.retrieveDataKeyPlainText(primaryKeyPath, dataKeyPath)
    val encrypt = new BigDLEncrypt()
    encrypt.init(AES_CBC_PKCS5PADDING, ENCRYPT, dataKeyPlaintext)
    encrypt.doFinal(csvFile, encryptedFilePath)

    // init a DataFrame for test
    val spark: SparkSession = SparkSession.builder()
      .master("local[1]").appName("initData")
      .getOrCreate()
    val data = Seq(("Java", "20000"), ("Python", "100000"), ("Scala", "3000"))
    df = spark.createDataFrame(data).toDF("language", "user")
    dataContent = df.orderBy("language").collect()
      .map(v => s"${v.get(0)},${v.get(1)}").mkString("\n")

    // init PPMLContext
    sc = ppmlContextPython.createPPMLContext(appName, ppmlArgs.asJava)
  }

  override def afterAll(): Unit = {
    val csvFile = new File(dir + "people.csv")
    val primaryKey = new File(dir + "primaryKey")
    val dataKey = new File(dir + "dataKey")
    val encryptPath = new File(dir + "encrypted")
    val writeOutput = new File(dir + "output")
    val parquetPath = new File(dir + "parquet")
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
    if (parquetPath.isDirectory) {
      deleteDir(parquetPath)
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
    val encryptedDataFrameReader = ppmlContextPython.read(sc, cryptoMode)
    ppmlContextPython.option(encryptedDataFrameReader, "header", "true")
    val dfFromCsv = ppmlContextPython.csv(encryptedDataFrameReader, path)

    Log4Error.invalidOperationError(dfFromCsv.count() == 8,
        "record count should be 8")
  }

  def write(df: DataFrame, encryptMode: String): Unit = {
    var outputDir = "output/"
    if (encryptMode == "AES/CBC/PKCS5Padding") {
      outputDir = outputDir + "encrypt"
    } else {
      outputDir = outputDir + "plain"
    }
    val path = dir + outputDir

    val encryptedDataFrameWriter = ppmlContextPython.write(sc, df, encryptMode)
    ppmlContextPython.mode(encryptedDataFrameWriter, "overwrite")
    ppmlContextPython.option(encryptedDataFrameWriter, "header", true)
    ppmlContextPython.csv(encryptedDataFrameWriter, path)
  }

  test("init PPMLContext with app name") {
    ppmlContextPython.createPPMLContext(appName)
  }

  test("init PPMLContext with app name & args") {
    ppmlContextPython.createPPMLContext(appName, ppmlArgs.asJava)
  }

  test("read plain text csv file") {
    val cryptoMode = "plain_text"
    val path = dir + "people.csv"

    read(cryptoMode, path)
  }

  test("read encrypted csv file") {
    val cryptoMode = "AES/CBC/PKCS5Padding"
    val path = dir + "encrypted/people.csv"

    read(cryptoMode, path)
  }

  test(" write plain text csv file") {
    write(df, "plain_text")
  }

  test(" write encrypted csv file") {
    write(df, "AES/CBC/PKCS5Padding")
  }

  test("write and read plain parquet file") {
    // write a parquet file
    val path = dir + "parquet/plain-parquet"
    val encryptedDataFrameWriter = ppmlContextPython.write(sc, df, "plain_text")
    ppmlContextPython.mode(encryptedDataFrameWriter, "overwrite")
    ppmlContextPython.parquet(encryptedDataFrameWriter, path)

    // read a parquet file
    val encryptedDataFrameReader = ppmlContextPython.read(sc, "plain_text")
    val parquetDF = ppmlContextPython.parquet(encryptedDataFrameReader, path)

    val data = parquetDF.orderBy("language").collect()
      .map(v => s"${v.get(0)},${v.get(1)}").mkString("\n")
    Log4Error.invalidOperationError(data == dataContent,
      "current data is\n" + data + "\n" +
        "data should be\n" + dataContent)
  }

  test("write and read encrypted parquet file") {
    // write a parquet file
    val path = dir + "parquet/en-parquet"
    val encryptedDataFrameWriter = ppmlContextPython.write(sc, df, "AES_GCM_CTR_V1")
    ppmlContextPython.mode(encryptedDataFrameWriter, "overwrite")
    ppmlContextPython.parquet(encryptedDataFrameWriter, path)

    // read a parquet file
    val encryptedDataFrameReader = ppmlContextPython.read(sc, "AES_GCM_CTR_V1")
    val parquetDF = ppmlContextPython.parquet(encryptedDataFrameReader, path)

    val data = parquetDF.orderBy("language").collect()
      .map(v => s"${v.get(0)},${v.get(1)}").mkString("\n")
    Log4Error.invalidOperationError(data == dataContent,
      "current data is\n" + data + "\n" +
        "data should be\n" + dataContent)
  }

}
