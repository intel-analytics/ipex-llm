/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
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

import java.io.{File, RandomAccessFile}
import java.nio.charset.StandardCharsets
import java.util.Base64

/**
 * A test suite that tests parquet modular encryption usage.
 */
class ParquetEncryptionSpec extends DataFrameHelper {
  override val repeatedNum = 100

  private val encoder = Base64.getEncoder
  private val footerKey =
    encoder.encodeToString("0123456789012345".getBytes(StandardCharsets.UTF_8))
  private val key1 = encoder.encodeToString("1234567890123450".getBytes(StandardCharsets.UTF_8))
  private val key2 = encoder.encodeToString("1234567890123451".getBytes(StandardCharsets.UTF_8))

  val ppmlArgs = Map(
    "spark.bigdl.kms.simple.id" -> appid,
    "spark.bigdl.kms.simple.key" -> appkey,
    "spark.bigdl.kms.key.primary" -> primaryKeyPath,
    "spark.bigdl.kms.key.data" -> dataKeyPath
  )
  val conf = new SparkConf().setMaster("local[4]")
//  conf.set("parquet.crypto.factory.class",
//    "org.apache.parquet.crypto.keytools.PropertiesDrivenCryptoFactory")
//  conf.set("parquet.encryption.kms.client.class",
//    "org.apache.parquet.crypto.keytools.mocks.InMemoryKMS")
//  conf.set("parquet.encryption.key.list",
//    s"footerKey: ${footerKey}, key1: ${key1}, key2: ${key2}")
  val sc = PPMLContext.initPPMLContext(conf, "SimpleQuery", ppmlArgs)
  val sparkSession = sc.getSparkSession()
  sparkSession.sparkContext.hadoopConfiguration.set("parquet.crypto.factory.class",
    "org.apache.parquet.crypto.keytools.PropertiesDrivenCryptoFactory")
  sparkSession.sparkContext.hadoopConfiguration.set("parquet.encryption.kms.client.class",
    "org.apache.parquet.crypto.keytools.mocks.InMemoryKMS")
  sparkSession.sparkContext.hadoopConfiguration.set("parquet.encryption.key.list",
    s"footerKey: ${footerKey}, key1: ${key1}, key2: ${key2}")
  import sparkSession.implicits._

  "SPARK-34990: Write and read an encrypted parquet" should "work" in {
    val input = Seq((1, 22, 333))
    val inputDF = input.toDF("a", "b", "c")
    val parquetDir = new File(dir, "parquet").getCanonicalPath
    inputDF.write
      .option("parquet.encryption.column.keys", "key1: a, b; key2: c")
      .option("parquet.encryption.footer.key", "footerKey")
      .parquet(parquetDir)

    verifyParquetEncrypted(parquetDir)

    val parquetDF = sparkSession.read.parquet(parquetDir)
    assert(parquetDF.inputFiles.nonEmpty)
    val readDataset = parquetDF.select("a", "b", "c")
    val result = readDataset.collect()
    result.length should be (1)
    result(0).get(0) should be (input.head._1)
    result(0).get(1) should be (input.head._2)
    result(0).get(2) should be (input.head._3)
  }

  "SPARK-37117: Can't read files in Parquet encryption external key material mode" should "work" in {
    val input = Seq((1, 22, 333))
    val inputDF = input.toDF("a", "b", "c")
    val parquetDir = new File(dir, "parquet").getCanonicalPath
    inputDF.write
      .option("parquet.encryption.column.keys", "key1: a, b; key2: c")
      .option("parquet.encryption.footer.key", "footerKey")
      .parquet(parquetDir)

    val parquetDF = sparkSession.read.parquet(parquetDir)
    assert(parquetDF.inputFiles.nonEmpty)
    val readDataset = parquetDF.select("a", "b", "c")
    val result = readDataset.collect()
    result.length should be (1)
    result(0).get(0) should be (input.head._1)
    result(0).get(1) should be (input.head._2)
    result(0).get(2) should be (input.head._3)
  }

  "read from plain csv with header" should "work" in {
    val enParquetPath = dir + "/en-parquet"
    val plainParquetPath = dir + "/plain-parquet"
    val df = sc.read(cryptoMode = PLAIN_TEXT)
      .option("header", "true").csv(plainFileName)
    val d = df.schema.map(_.name).mkString(",") + "\n" +
      df.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d + "\n" should be (data)
    df.write
      .mode("overwrite")
      .option("parquet.encryption.column.keys", "key1: name, age; key2: job")
      .option("parquet.encryption.footer.key", "footerKey")
      .parquet(enParquetPath)
    df.write
      .mode("overwrite")
      .parquet(plainParquetPath)

    val df2 = sc.getSparkSession().read.parquet(enParquetPath)
    val d2 = df2.schema.map(_.name).mkString(",") + "\n" +
      df2.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d2 + "\n" should be (data)

    val df3 = sc.getSparkSession().read.parquet(enParquetPath)
    val d3 = df3.schema.map(_.name).mkString(",") + "\n" +
      df3.collect().map(v => s"${v.get(0)},${v.get(1)},${v.get(2)}").mkString("\n")
    d3 + "\n" should be (data)
  }

  /**
   * Verify that the directory contains an encrypted parquet in
   * encrypted footer mode by means of checking for all the parquet part files
   * in the parquet directory that their magic string is PARE, as defined in the spec:
   * https://github.com/apache/parquet-format/blob/master/Encryption.md#54-encrypted-footer-mode
   */
  private def verifyParquetEncrypted(parquetDir: String): Unit = {
    val parquetPartitionFiles = getListOfParquetFiles(new File(parquetDir))
    assert(parquetPartitionFiles.size >= 1)
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
      assert(magicString == stringRead)
    }
  }

  private def getListOfParquetFiles(dir: File): List[File] = {
    dir.listFiles.filter(_.isFile).toList.filter { file =>
      file.getName.endsWith("parquet")
    }
  }
}
