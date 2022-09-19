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
import com.intel.analytics.bigdl.ppml.crypto.{AES_GCM_CTR_V1, PLAIN_TEXT}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

import java.io.{File, RandomAccessFile}
import java.nio.charset.StandardCharsets
import java.util.Base64

/**
 * A test suite that tests parquet modular encryption usage.
 */
class ParquetEncryptionSpec extends DataFrameHelper {
  val (plainFileName, encryptFileName, data, dataKeyPlaintext) = generateCsvData()

  val ppmlArgs = Map(
    "spark.bigdl.kms.simple.id" -> appid,
    "spark.bigdl.kms.simple.key" -> apikey,
    "spark.bigdl.kms.key.primary" -> primaryKeyPath,
    "spark.bigdl.kms.key.data" -> dataKeyPath
  )
  val conf = new SparkConf().setMaster("local[4]")
  val sc = PPMLContext.initPPMLContext(conf, "SimpleQuery", ppmlArgs)

  "write/read encrypted parquet format" should "work" in {
    val enParquetPath = dir + "/en-parquet"
    val df = sc.read(cryptoMode = PLAIN_TEXT)
      .option("header", "true").csv(plainFileName)
    val d = df.schema.map(_.name).mkString(",") + "\n" +
      df.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d + "\n" should be (data)
    sc.write(df, AES_GCM_CTR_V1)
      .mode("overwrite")
      .parquet(enParquetPath)

    val df2 = sc.read(AES_GCM_CTR_V1).parquet(enParquetPath)
    val d2 = df2.schema.map(_.name).mkString(",") + "\n" +
      df2.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d2 + "\n" should be (data)
  }

  "SPARK-34990: Write and read an encrypted parquet" should "work" in {
    val sparkSession = sc.getSparkSession()
    import sparkSession.implicits._
    val input = Seq((1, 22, 333))
    val inputDF = input.toDF("a", "b", "c")
    val parquetDir = new File(dir, "parquet").getCanonicalPath
    sc.write(inputDF, AES_GCM_CTR_V1)
      .parquet(parquetDir)

    verifyParquetEncrypted(parquetDir)

    val parquetDF = sparkSession.read.parquet(parquetDir)
    parquetDF.inputFiles.nonEmpty should be (true)
    val readDataset = parquetDF.select("a", "b", "c")
    val result = readDataset.collect()
    result.length should be (1)
    result(0).get(0) should be (input.head._1)
    result(0).get(1) should be (input.head._2)
    result(0).get(2) should be (input.head._3)
  }

  "write/read parquet" should "work" in {
    val plainParquetPath = dir + "/plain-parquet"
    val df = sc.read(cryptoMode = PLAIN_TEXT)
      .option("header", "true").csv(plainFileName)
    val d = df.schema.map(_.name).mkString(",") + "\n" +
      df.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d + "\n" should be (data)
    sc.write(df, PLAIN_TEXT)
      .mode("overwrite")
      .parquet(plainParquetPath)

    val df2 = sc.read(PLAIN_TEXT).parquet(plainParquetPath)
    val d2 = df2.schema.map(_.name).mkString(",") + "\n" +
      df2.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d2 + "\n" should be (data)
  }

  /**
   * Verify that the directory contains an encrypted parquet in
   * encrypted footer mode by means of checking for all the parquet part files
   * in the parquet directory that their magic string is PARE, as defined in the spec:
   * https://github.com/apache/parquet-format/blob/master/Encryption.md#54-encrypted-footer-mode
   */
  private def verifyParquetEncrypted(parquetDir: String): Unit = {
    val parquetPartitionFiles = getListOfParquetFiles(new File(parquetDir))
    parquetPartitionFiles.size >= 1 should be (true)
    parquetPartitionFiles.foreach { parquetFile =>
      val magicString = "PARE"
      val magicStringLength = magicString.length()
      val byteArray = new Array[Byte](magicStringLength)
      val randomAccessFile = new RandomAccessFile(parquetFile, "r")
      try {
        randomAccessFile.read(byteArray, 0, magicStringLength)
      } finally {
        randomAccessFile.close()
      }
      val stringRead = new String(byteArray, StandardCharsets.UTF_8)
      magicString should be (stringRead)
    }
  }

  private def getListOfParquetFiles(dir: File): List[File] = {
    dir.listFiles.filter(_.isFile).toList.filter { file =>
      file.getName.endsWith("parquet")
    }
  }
}
