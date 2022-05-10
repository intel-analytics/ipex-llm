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

package com.intel.analytics.bigdl.ppml

import com.intel.analytics.bigdl.dllib.NNContext.{checkScalaVersion, checkSparkVersion, createSparkConf, initConf, initNNContext}
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.crypto.CryptoMode.CryptoMode
import com.intel.analytics.bigdl.ppml.crypto.{CryptoMode, EncryptRuntimeException, FernetEncrypt}
import com.intel.analytics.bigdl.ppml.utils.Supportive
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.input.PortableDataStream
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, DataFrameReader, DataFrameWriter, Row, SparkSession}
import com.intel.analytics.bigdl.ppml.kms.{EHSMKeyManagementService, KMS_CONVENTION, KeyManagementService, SimpleKeyManagementService}
import com.intel.analytics.bigdl.ppml.crypto.dataframe.EncryptedDataFrameReader

import java.nio.file.Paths

/**
 * PPMLContext who wraps a SparkSession and provides read functions to
 * read encrypted data files to plain-text RDD or DataFrame, also provides
 * write functions to save DataFrame to encrypted data files.
 * @param kms
 * @param sparkSession
 */
class PPMLContext protected(kms: KeyManagementService, sparkSession: SparkSession) {

  protected var dataKeyPlainText: String = ""

  /**
   * Load keys from a local file system.
   * @param primaryKeyPath
   * @param dataKeyPath
   * @return
   */
  def loadKeys(primaryKeyPath: String, dataKeyPath: String): this.type = {
    dataKeyPlainText = kms.retrieveDataKeyPlainText(
      Paths.get(primaryKeyPath).toString, Paths.get(dataKeyPath).toString)
    this
  }

  /**
   * Read data files into RDD[String]
   * @param path data file path
   * @param minPartitions min partitions
   * @param cryptoMode crypto mode, such as PLAIN_TEXT or AES_CBC_PKCS5PADDING
   * @return
   */
  def textFile(path: String,
               minPartitions: Int = sparkSession.sparkContext.defaultMinPartitions,
               cryptoMode: CryptoMode = CryptoMode.PLAIN_TEXT): RDD[String] = {
    cryptoMode match {
      case CryptoMode.PLAIN_TEXT =>
        sparkSession.sparkContext.textFile(path, minPartitions)
      case CryptoMode.AES_CBC_PKCS5PADDING =>
        PPMLContext.textFile(sparkSession.sparkContext, path, dataKeyPlainText, minPartitions)
      case _ =>
        throw new IllegalArgumentException("unknown EncryptMode " + cryptoMode.toString)
    }
  }

  /**
   * Interface for loading data in external storage to Dataset.
   * @param cryptoMode crypto mode, such as PLAIN_TEXT or AES_CBC_PKCS5PADDING
   * @return a EncryptedDataFrameReader
   */
  def read(cryptoMode: CryptoMode): EncryptedDataFrameReader = {
    new EncryptedDataFrameReader(sparkSession, cryptoMode, dataKeyPlainText)
  }

  /**
   * Interface for saving the content of the non-streaming Dataset out into external storage.
   * @param dataFrame dataframe to save.
   * @param cryptoMode crypto mode, such as PLAIN_TEXT or AES_CBC_PKCS5PADDING
   * @return a DataFrameWriter[Row]
   */
  def write(dataFrame: DataFrame, cryptoMode: CryptoMode): DataFrameWriter[Row] = {
    cryptoMode match {
      case CryptoMode.PLAIN_TEXT =>
        dataFrame.write
      case CryptoMode.AES_CBC_PKCS5PADDING =>
        PPMLContext.write(sparkSession, dataKeyPlainText, dataFrame)
      case _ =>
        throw new IllegalArgumentException("unknown EncryptMode " + cryptoMode.toString)
    }
  }
}

object PPMLContext{
  private[bigdl] def registerUDF(spark: SparkSession,
                  dataKeyPlaintext: String) = {
    val bcKey = spark.sparkContext.broadcast(dataKeyPlaintext)
    val convertCase = (x: String) => {
      val fernetCryptos = new FernetEncrypt()
      new String(fernetCryptos.encryptBytes(x.getBytes, bcKey.value))
    }
    spark.udf.register("convertUDF", convertCase)
  }

  private[bigdl] def textFile(sc: SparkContext,
               path: String,
               dataKeyPlaintext: String,
               minPartitions: Int = -1): RDD[String] = {
    Log4Error.invalidInputError(dataKeyPlaintext != "", "dataKeyPlainText should not be empty, please loadKeys first.")
    val data: RDD[(String, PortableDataStream)] = if (minPartitions > 0) {
      sc.binaryFiles(path, minPartitions)
    } else {
      sc.binaryFiles(path)
    }
    val fernetCryptos = new FernetEncrypt
    data.mapPartitions { iterator => {
      Supportive.logger.info("Decrypting bytes with JavaAESCBC...")
      fernetCryptos.decryptBigContent(iterator, dataKeyPlaintext)
    }}.flatMap(_.split("\n"))
  }

  private[bigdl] def write(sparkSession: SparkSession,
               dataKeyPlaintext: String,
               dataFrame: DataFrame): DataFrameWriter[Row] = {
    val tableName = "ppml_save_table"
    dataFrame.createOrReplaceTempView(tableName)
    PPMLContext.registerUDF(sparkSession, dataKeyPlaintext)
    // Select all and encrypt columns.
    val convertSql = "select " + dataFrame.schema.map(column =>
      "convertUDF(" + column.name + ") as " + column.name).mkString(", ") +
      " from " + tableName
    val df = sparkSession.sql(convertSql)
    df.write
  }

  def initPPMLContext(appName: String): PPMLContext = {
    initPPMLContext(null, appName)
  }

  def initPPMLContext(conf: SparkConf): PPMLContext = {
    initPPMLContext(conf)
  }

  /**
   * init ppml context with app name and ppml args
   * @param appName the name of this Application
   * @param ppmlArgs ppml arguments in a Map
   * @return a PPMLContext
   */
  def initPPMLContext(
        appName: String,
        ppmlArgs: Map[String, String]): PPMLContext = {
    initPPMLContext(null, appName, ppmlArgs)
  }

  /**
   * init ppml context with app name, SparkConf and ppml args
   * @param sparkConf a SparkConf
   * @param appName the name of this Application
   * @param ppmlArgs ppml arguments in a Map
   * @return a PPMLContext
   */
  def initPPMLContext(
        sparkConf: SparkConf,
        appName: String,
        ppmlArgs: Map[String, String]): PPMLContext = {
    val conf = createSparkConf(sparkConf)
    ppmlArgs.foreach{arg =>
      conf.set(arg._1, arg._2)
    }
    initPPMLContext(conf, appName)
  }

  /**
   * init ppml context with app name, SparkConf
   * @param sparkConf a SparkConf, ppml arguments are passed by this sparkconf.
   * @param appName the name of this Application
   * @param ppmlArgs ppml arguments in a Map
   * @return a PPMLContext
   */
  def initPPMLContext(sparkConf: SparkConf, appName: String): PPMLContext = {
    val conf = createSparkConf(sparkConf)
    val sc = initNNContext(conf, appName)
    val sparkSession: SparkSession = SparkSession.builder().getOrCreate()
    val kmsType = conf.get("spark.bigdl.kms.type", defaultValue = "SimpleKeyManagementService")
    val kms = kmsType match {
      case KMS_CONVENTION.MODE_EHSM_KMS =>
        val ip = conf.get("spark.bigdl.kms.ehs.ip", defaultValue = "0.0.0.0")
        val port = conf.get("spark.bigdl.kms.ehs.port", defaultValue = "5984")
        val appId = conf.get("spark.bigdl.kms.ehs.id", defaultValue = "ehsmAPPID")
        val appKey = conf.get("spark.bigdl.kms.ehs.key", defaultValue = "ehsmAPPKEY")
        new EHSMKeyManagementService(ip, port, appId, appKey)
      case KMS_CONVENTION.MODE_SIMPLE_KMS =>
        val id = conf.get("spark.bigdl.kms.simple.id", defaultValue = "simpleAPPID")
        // println(id + "=-------------------")
        val key = conf.get("spark.bigdl.kms.simple.key", defaultValue = "simpleAPPKEY")
        // println(key + "=-------------------")
        new SimpleKeyManagementService(id, key)
      case _ =>
        throw new EncryptRuntimeException("Wrong kms type")
    }
    val ppmlSc = new PPMLContext(kms, sparkSession)
    if (conf.contains("spark.bigdl.kms.key.primary")){
      Log4Error.invalidInputError(conf.contains("spark.bigdl.kms.key.data"),
        "Data key not found, please provide" +
        " both spark.bigdl.kms.key.primary and spark.bigdl.kms.key.data.")
      val primaryKey = conf.get("spark.bigdl.kms.key.primary")
      val dataKey = conf.get("spark.bigdl.kms.key.data")
      ppmlSc.loadKeys(primaryKey, dataKey)
    }
    ppmlSc
  }

}
