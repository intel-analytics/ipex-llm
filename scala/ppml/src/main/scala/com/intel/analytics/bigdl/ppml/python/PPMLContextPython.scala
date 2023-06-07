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

import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.dataframe.{EncryptedDataFrameReader, EncryptedDataFrameWriter}
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, BigDLEncrypt, CryptoMode, ENCRYPT}
import com.intel.analytics.bigdl.ppml.kms.SimpleKeyManagementService
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.{Logger, LoggerFactory}

import java.util
import scala.collection.JavaConversions._




object PPMLContextPython {
  def ofFloat: PPMLContextPython[Float] = new PPMLContextPython[Float]()

  def ofDouble: PPMLContextPython[Double] = new PPMLContextPython[Double]()
}

class PPMLContextPython[T]() {
  val logger: Logger = LoggerFactory.getLogger(getClass)

  def createPPMLContext(sparkSession: SparkSession): PPMLContext = {
    logger.debug("createPPMLContext with SparkSession" + "confs:\n" +
      sparkSession.sparkContext.getConf.getAll.mkString("Array(", ", ", ")"))
    PPMLContext.initPPMLContext(sparkSession)
  }

  def read(sc: PPMLContext, cryptoModeStr: String,
    primaryKeyName: String = "defaultKey"): EncryptedDataFrameReader = {
    logger.debug("read file with crypto mode " + cryptoModeStr)
    val cryptoMode = CryptoMode.parse(cryptoModeStr)
    sc.read(cryptoMode, primaryKeyName)
  }

  def write(sc: PPMLContext, dataFrame: DataFrame, cryptoModeStr: String,
    primaryKeyName: String = "defaultKey"): EncryptedDataFrameWriter = {
    logger.debug("write file with crypt mode " + cryptoModeStr)
    val cryptoMode = CryptoMode.parse(cryptoModeStr)
    sc.write(dataFrame, cryptoMode, primaryKeyName)
  }

  def textFile(sc: PPMLContext, path: String,
    minPartitions: Int, cryptoModeStr: String,
    primaryKeyName: String = "defaultKey"): RDD[String] = {
    val cryptoMode = CryptoMode.parse(cryptoModeStr)
    sc.textFile(path, minPartitions, cryptoMode, primaryKeyName)
  }

  def sql(sc: PPMLContext, sqlText: String = "defaultKey"): DataFrame = {
    logger.debug("running spark sql " + sqlText)
    sc.sql(sqlText)
  }

  /**
   * EncryptedDataFrameReader method
   */

  def option(encryptedDataFrameReader: EncryptedDataFrameReader,
             key: String, value: String): EncryptedDataFrameReader = {
    encryptedDataFrameReader.option(key, value)
  }
  import net.razorvine.pickle.objects.ClassDict

  def schema(encryptedDataFrameReader: EncryptedDataFrameReader,
             value: ClassDict): EncryptedDataFrameReader = {
    val hashMap: util.HashMap[String, Object] = value.asInstanceOf[util.HashMap[String, Object]]
    val resList = new util.LinkedList[StructField]()
    // get fields list
    val fieldList = hashMap.get("fields").asInstanceOf[java.util.List[_]]
    if (fieldList == null) {
      throw new IllegalArgumentException("schema cann't be null")
    }
    for ( map <- fieldList) {
      val fieldMap = map.asInstanceOf[util.HashMap[String, Object]]
      val dataTypeMap = fieldMap.get("dataType").asInstanceOf[util.HashMap[String, Object]]
      val structType = dataTypeMap.get("__class__").toString
      val fieldName = fieldMap.get("name").toString
      val nullable = fieldMap.get("nullable").toString.toLowerCase().equals("true")
      val fieldType = structType match {
        case "pyspark.sql.types.ByteType" => ByteType
        case "pyspark.sql.types.ShortType" => ShortType
        case "pyspark.sql.types.IntegerType" => IntegerType
        case "pyspark.sql.types.LongType" => LongType
        case "pyspark.sql.types.FloatType" => FloatType
        case "pyspark.sql.types.DoubleType" => DoubleType
        case "pyspark.sql.types.BooleanType" => BooleanType
        case "pyspark.sql.types.StringType" => StringType
        case _ =>
          throw new IllegalArgumentException(s"Unsupported data type for field $structType")
      }
      val structField = StructField(fieldName, fieldType, nullable = nullable)
      resList.add(structField)
    }
    val schema = StructType(resList)
    println("Schema: " + schema)
    encryptedDataFrameReader.schema(schema)
  }

  def csv(encryptedDataFrameReader: EncryptedDataFrameReader, path: String): DataFrame = {
    logger.debug("read csv file from path: " + path)
    encryptedDataFrameReader.csv(path)
  }

  def parquet(encryptedDataFrameReader: EncryptedDataFrameReader, path: String): DataFrame = {
    logger.debug("read parquet file from path: " + path)
    encryptedDataFrameReader.parquet(path)
  }

  def json(encryptedDataFrameReader: EncryptedDataFrameReader, path: String): DataFrame = {
    logger.debug("read json file from path: " + path)
    encryptedDataFrameReader.json(path)
  }

  /**
   * EncryptedDataFrameWriter method
   */

  def option(encryptedDataFrameWriter: EncryptedDataFrameWriter,
             key: String, value: String): EncryptedDataFrameWriter = {
    encryptedDataFrameWriter.option(key, value)
  }

  def option(encryptedDataFrameWriter: EncryptedDataFrameWriter,
             key: String, value: Boolean): EncryptedDataFrameWriter = {
    encryptedDataFrameWriter.option(key, value)
  }

  def mode(encryptedDataFrameWriter: EncryptedDataFrameWriter,
           mode: String): EncryptedDataFrameWriter = {
    encryptedDataFrameWriter.mode(mode)
  }

  def csv(encryptedDataFrameWriter: EncryptedDataFrameWriter, path: String): Unit = {
    encryptedDataFrameWriter.csv(path)
  }

  def parquet(encryptedDataFrameWriter: EncryptedDataFrameWriter, path: String): Unit = {
    encryptedDataFrameWriter.parquet(path)
  }

  def json(encryptedDataFrameWriter: EncryptedDataFrameWriter, path: String): Unit = {
    encryptedDataFrameWriter.json(path)
  }

  /**
   * support for test
   */

  def initKeys(appId: String, apiKey: String,
               primaryKeyPath: String): SimpleKeyManagementService = {
    val kms = SimpleKeyManagementService.apply(appId, apiKey)
    kms.retrievePrimaryKey(primaryKeyPath)
    kms
  }

  def generateEncryptedFile(kms: SimpleKeyManagementService, primaryKeyPath: String,
                            dataKeyPath: String, input: String, output: String): Unit = {
    val dataKeyPlaintext = kms.retrieveDataKeyPlainText(primaryKeyPath, dataKeyPath)
    val encrypt = new BigDLEncrypt()
    encrypt.init(AES_CBC_PKCS5PADDING, ENCRYPT, dataKeyPlaintext)
    encrypt.doFinal(input, output)
  }

}
