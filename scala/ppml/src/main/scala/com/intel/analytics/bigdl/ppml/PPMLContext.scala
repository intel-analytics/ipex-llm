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
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, BigDLEncrypt, Crypto, CryptoMode, DECRYPT, ENCRYPT, EncryptRuntimeException, PLAIN_TEXT}
import com.intel.analytics.bigdl.ppml.utils.Supportive
import com.intel.analytics.bigdl.ppml.kms.common.{KeyLoader, KeyLoaderManagement}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.input.PortableDataStream
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, DataFrameReader, DataFrameWriter, Row, SparkSession}
import com.intel.analytics.bigdl.ppml.kms.{AzureKeyManagementService, EHSMKeyManagementService, KMS_CONVENTION,
KeyManagementService, SimpleKeyManagementService, BigDLKeyManagementService}
import com.intel.analytics.bigdl.ppml.crypto.dataframe.{EncryptedDataFrameReader, EncryptedDataFrameWriter}
import org.apache.hadoop.fs.Path

/**
 * PPMLContext who wraps a SparkSession and provides read functions to
 * read encrypted data files to plain-text RDD or DataFrame, also provides
 * write functions to save DataFrame to encrypted data files.
 */
class PPMLContext {
  protected val keyLoaderManagement = new KeyLoaderManagement
  protected var sparkSession: SparkSession = null
  protected var defaultKey: String = ""

  /**
   * Read data files into RDD[String]
   * @param path data file path
   * @param minPartitions min partitions
   * @param cryptoMode crypto mode, such as PLAIN_TEXT or AES_CBC_PKCS5PADDING
   * @param kmsName for multi-data source/KMS mode, name of kms which this data source uses
   * @param primaryKeyName use the name-specific primary key to decrypt data key in file meata
   * @return
   */
  def textFile(path: String,
               minPartitions: Int = sparkSession.sparkContext.defaultMinPartitions,
               cryptoMode: CryptoMode = PLAIN_TEXT,
               primaryKeyName: String = "defaultKey"): RDD[String] = {
    cryptoMode match {
      case PLAIN_TEXT =>
        sparkSession.sparkContext.textFile(path, minPartitions)
      case _ =>
        val dataKeyPlainText = primaryKeyName match {
          case "" =>
            keyLoaderManagement.retrieveKeyLoader(defaultKey)
                               .retrieveDataKeyPlainText(path)
          case _ =>
            keyLoaderManagement.retrieveKeyLoader(primaryKeyName)
                               .retrieveDataKeyPlainText(path)
        }
        PPMLContext.textFile(sparkSession.sparkContext, path, dataKeyPlainText,
                             cryptoMode, minPartitions)
    }
  }

  /**
   * Interface for loading data in external storage to Dataset.
   * @param cryptoMode crypto mode, such as PLAIN_TEXT or AES_CBC_PKCS5PADDING
   * @param primaryKeyName use the name-specific primary key to decrypt data key in file meata
   * @return a EncryptedDataFrameReader
   */
  def read(cryptoMode: CryptoMode,
           primaryKeyName: String = ""): EncryptedDataFrameReader = {
      primaryKeyName match {
        case "" =>
          new EncryptedDataFrameReader(sparkSession, cryptoMode,
            defaultKey, keyLoaderManagement)
        case _ =>
          new EncryptedDataFrameReader(sparkSession, cryptoMode,
            primaryKeyName, keyLoaderManagement)
      }
  }

  /**
   * Interface for saving the content of the non-streaming Dataset out into external storage.
   * @param dataFrame dataframe to save.
   * @param cryptoMode crypto mode, such as PLAIN_TEXT or AES_CBC_PKCS5PADDING
   * @param primaryKeyName use the name-specific primary key to decrypt data key in file meata
   * @return a DataFrameWriter[Row]
   */
  def write(dataFrame: DataFrame,
            cryptoMode: CryptoMode,
            primaryKeyName: String = ""): EncryptedDataFrameWriter = {
      primaryKeyName match {
        case "" =>
          new EncryptedDataFrameWriter(sparkSession, dataFrame, cryptoMode,
            defaultKey, keyLoaderManagement)
        case _ =>
          new EncryptedDataFrameWriter(sparkSession, dataFrame, cryptoMode,
            primaryKeyName, keyLoaderManagement)
      }
  }

  /**
   * Get SparkSession from PPMLContext
   * @return SparkSession in PPMLContext
   */
  def getSparkSession(): SparkSession = {
    sparkSession
  }
}


object PPMLContext{
  private[bigdl] def registerUDF(
        spark: SparkSession,
        cryptoMode: CryptoMode,
        dataKeyPlaintext: String) = {
    val bcKey = spark.sparkContext.broadcast(dataKeyPlaintext)
    val convertCase = (x: String) => {
      val crypto = Crypto(cryptoMode)
      crypto.init(cryptoMode, ENCRYPT, dataKeyPlaintext)
      new String(crypto.doFinal(x.getBytes)._1)
    }
    spark.udf.register("convertUDF", convertCase)
  }

  private[bigdl] def textFile(sc: SparkContext,
               path: String,
               dataKeyPlaintext: String,
               cryptoMode: CryptoMode,
               minPartitions: Int = -1): RDD[String] = {
    Log4Error.invalidInputError(dataKeyPlaintext != "",
      "dataKeyPlainText should not be empty, please loadKeys first.")
    val data: RDD[(String, PortableDataStream)] = if (minPartitions > 0) {
      sc.binaryFiles(path, minPartitions)
    } else {
      sc.binaryFiles(path)
    }
    data.mapPartitions { iterator => {
      Supportive.logger.info("Decrypting bytes with JavaAESCBC...")
      iterator.flatMap{dataStream =>
        val inputDataStream = dataStream._2.open()
        val crypto = Crypto(cryptoMode)
        val (encryptedDataKey, initializationVector) = crypto.getHeader(inputDataStream)
        crypto.init(cryptoMode, DECRYPT, dataKeyPlaintext)
        crypto.verifyHeader(initializationVector)
        crypto.decryptBigContent(inputDataStream)
      }
    }}
  }

  private[bigdl] def write(
        sparkSession: SparkSession,
        cryptoMode: CryptoMode,
        dataKeyPlaintext: String,
        dataFrame: DataFrame): DataFrameWriter[Row] = {
    val tableName = "ppml_save_table"
    dataFrame.createOrReplaceTempView(tableName)
    PPMLContext.registerUDF(sparkSession, cryptoMode, dataKeyPlaintext)
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
   * init ppml context with an existed SparkSession
   * @param sparkSession a SparkSession
   * @return a PPMLContext
   */
  def initPPMLContext(sparkSession: SparkSession): PPMLContext = {
    val conf = sparkSession.sparkContext.getConf
    Log4Error.invalidInputError(conf.contains("spark.hadoop.io.compression.codecs"),
        "spark.hadoop.io.compression.codecs not found!" +
        "If you want to init PPMLContext with an existing SparkSession, " +
        "must set the property before creating SparkSession!")
    Log4Error.invalidInputError(
      conf.get("spark.hadoop.io.compression.codecs")
      == "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec",
      "If you want to init PPMLContext with an existing SparkSession, " +
      "spark.hadoop.io.compression.codecs property must be set to" +
      "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec" +
      " before creating SparkSession!")
    val ppmlSc = loadPPMLContext(sparkSession)
    ppmlSc
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
    ppmlArgs.foreach { arg =>
      conf.set(arg._1, arg._2)
    }
    initPPMLContext(conf, appName)
  }

  /**
   * init ppml context with app name, SparkConf
   * @param sparkConf a SparkConf, ppml arguments are passed by this sparkconf.
   * @param appName the name of this Application
   * @return a PPMLContext
   */
  def initPPMLContext(sparkConf: SparkConf, appName: String): PPMLContext = {
    val conf = createSparkConf(sparkConf)
    conf.set("spark.hadoop.io.compression.codecs",
        "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec")
    val sc = initNNContext(conf, appName)
    val sparkSession: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val ppmlSc = loadPPMLContext(sparkSession)
    ppmlSc
  }

  /**
   * load ppml context from spark session juding whether multiple KMSs exist
   * @param sparkSession a SparkSession
   * @return a PPMLContext
   */
  def loadPPMLContext(sparkSession: SparkSession): PPMLContext = {
    val ppmlSc = new PPMLContext
    ppmlSc.sparkSession = sparkSession
    val conf = sparkSession.sparkContext.getConf
    sparkSession.sparkContext.hadoopConfiguration.set(
      "spark.bigdl.encryter.type",
      conf.get("spark.bigdl.encryter.type", BigDLEncrypt.COMMON))
    val primaryKeyNames = getPrimaryKeyNames(conf)
    primaryKeyNames.foreach{
      primaryKeyName => {
        if (conf.contains(s"spark.bigdl.primaryKey.$primaryKeyName.plainText")) {
          val primaryKeyPlainText = conf.get(
            s"spark.bigdl.primaryKey.$primaryKeyName.plainText")
          ppmlSc.keyLoaderManagement
                .addKeyLoader(primaryKeyName,
                              KeyLoader(false, "", null, primaryKeyPlainText))
        } else {
          Log4Error.invalidInputError(
            conf.contains(s"spark.bigdl.primaryKey.$primaryKeyName.material"),
                          s"spark.bigdl.primaryKey.$primaryKeyName.material not found.")
            val primaryKeyMaterial = conf.get(
              s"spark.bigdl.primaryKey.$primaryKeyName.material")
            val kms = loadKmsOfPrimaryKey(conf, primaryKeyName)
            ppmlSc.keyLoaderManagement
                  .addKeyLoader(primaryKeyName,
                                KeyLoader(true, primaryKeyMaterial, kms, ""))
         }
      }
    }
    if (ppmlSc.keyLoaderManagement.count == 1) ppmlSc.defaultKey = primaryKeyNames(0)
    ppmlSc
  }

  def getPrimaryKeyNames(conf: SparkConf): Array[String] = {
    val prefix = "spark.bigdl.primaryKey"
    val properties: Array[Tuple2[String, String]] = conf.getAllWithPrefix(prefix)
    val names = for { v <- properties } yield v._1.split('.')(1)
    names.distinct
  }

  def loadKmsOfPrimaryKey(conf: SparkConf, primaryKeyName: String): KeyManagementService = {
    val kmsType = conf.get(s"spark.bigdl.primaryKey.$primaryKeyName.kms.type",
      defaultValue = KMS_CONVENTION.MODE_SIMPLE_KMS)
    val kms = kmsType match {
      case KMS_CONVENTION.MODE_EHSM_KMS =>
        val ip = conf.get(s"spark.bigdl.primaryKey.$primaryKeyName.kms.ip")
        val port = conf.get(s"spark.bigdl.primaryKey.$primaryKeyName.kms.port")
        val appId = conf.get(s"spark.bigdl.primaryKey.$primaryKeyName.kms.appId")
        val apiKey = conf.get(s"spark.bigdl.primaryKey.$primaryKeyName.kms.apiKey")
        new EHSMKeyManagementService(ip, port, appId, apiKey)
      case KMS_CONVENTION.MODE_SIMPLE_KMS =>
        val appId = conf.get(s"spark.bigdl.primaryKey.$primaryKeyName.kms.appId",
                             defaultValue = "simpleAPPID")
        val apiKey = conf.get(s"spark.bigdl.primaryKey.$primaryKeyName.kms.apiKey",
                              defaultValue = "simpleAPIKEY")
        SimpleKeyManagementService(appId, apiKey)
      case KMS_CONVENTION.MODE_AZURE_KMS =>
        val vaultName = conf.get(s"spark.bigdl.primaryKey.$primaryKeyName.kms.vault")
        val clientId = conf.get(s"spark.bigdl.primaryKey.$primaryKeyName.kms.clientId")
        new AzureKeyManagementService(vaultName, clientId)
      case KMS_CONVENTION.MODE_BIGDL_KMS =>
        val ip = conf.get(s"spark.bigdl.primaryKey.$primaryKeyName.kms.ip")
        val port = conf.get(s"spark.bigdl.primaryKey.$primaryKeyName.kms.port")
        val userName = conf.get(s"spark.bigdl.primaryKey.$primaryKeyName.kms.user")
        val userToken = conf.get(s"spark.bigdl.primaryKey.$primaryKeyName.kms.token")
        new BigDLKeyManagementService(ip, port, userName, userToken)
      case _ =>
        throw new EncryptRuntimeException("Wrong kms type")
    }
    kms
  }
}
