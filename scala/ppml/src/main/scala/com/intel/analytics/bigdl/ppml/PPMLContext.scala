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
import com.intel.analytics.bigdl.ppml.utils.{Supportive, DataSource, DataSink}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.input.PortableDataStream
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, DataFrameReader, DataFrameWriter, Row, SparkSession}
import com.intel.analytics.bigdl.ppml.kms.{AzureKeyManagementService, EHSMKeyManagementService, KMS_CONVENTION,
KeyManagementService, SimpleKeyManagementService, BigDLKeyManagementService, KMSManagement}
import com.intel.analytics.bigdl.ppml.crypto.dataframe.{EncryptedDataFrameReader, EncryptedDataFrameWriter}
import org.apache.hadoop.fs.Path

/**
 * PPMLContext who wraps a SparkSession and provides read functions to
 * read encrypted data files to plain-text RDD or DataFrame, also provides
 * write functions to save DataFrame to encrypted data files.
 * @param kms for single-data source/kms mode
 * @param sparkSession
 */
class PPMLContext protected(kms: KeyManagementService = null, sparkSession: SparkSession = null) {

  protected var dataSourceKeyPlainText: String = ""
  protected var dataSinkKeyPlainText: String = ""
  protected var dataSink: DataSink = null
  protected val kmsManagement = new KMSManagement

  /**
   * Load unique key for both read and write from a local file system for single-kms/dataSource mode
   * @param primaryKeyCertificate
   * @param dataKeyCertificate
   * @return
   */
  def loadKeys(primaryKeyCertificate: String, dataKeyCertificate: String): this.type = {
    dataSourceKeyPlainText = getDataKeyPlainTextFromKMS(
        primaryKeyCertificate,
        dataKeyCertificate,
        this.kms)
    dataSinkKeyPlainText = dataSourceKeyPlainText
    sparkSession.sparkContext.hadoopConfiguration.set("bigdl.kms.dataSource.plaintextkey", dataSourceKeyPlainText)
    sparkSession.sparkContext.hadoopConfiguration.set("bigdl.kms.dataSink.plaintextkey", dataSinkKeyPlainText)
    this
  }

  /**
   * Load data source (read) key from a local file system for multi-kms/dataSource mode
   * @param primaryKeyCertificate
   * @param dataKeyCertificate
   * @param dataSourceName name of dataSource which dataKey is loaded for 
   * @return
   */
  def loadDataSourceKey(dataSource: DataSource): this.type = {
    val kms = kmsManagement.getKMS(dataSource.kmsName)
    dataSourceKeyPlainText = getDataKeyPlainTextFromKMS(dataSource.primaryKey, dataSource.dataKey, kms)
    sparkSession.sparkContext.hadoopConfiguration.set(
        "bigdl.kms.dataSource." + dataSourceName + ".plaintextkey",
        dataSourceKeyPlainText)
    this
  }

  /**
   * Load data sink (write) key from a local file system for multi-kms/dataSource mode
   * @param primaryKeyCertificate
   * @param dataKeyCertificate
   * @param dataSourceName name of dataSource which dataKey is loaded for 
   * @return
   */
  def loadDataSinkKey(): this.type = {
    val kms = kmsManagement.getKMS(dataSink.kmsName)
    dataSinkKeyPlainText = getDataKeyPlainTextFromKMS(dataSink.primaryKey, dataSink.dataKey, kms)
    sparkSession.sparkContext.hadoopConfiguration.set(
        "bigdl.kms.dataSink.plaintextkey",
        dataSinkKeyPlainText)
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
               cryptoMode: CryptoMode = PLAIN_TEXT): RDD[String] = {
    cryptoMode match {
      case PLAIN_TEXT =>
        sparkSession.sparkContext.textFile(path, minPartitions)
      case _ =>
        PPMLContext.textFile(sparkSession.sparkContext, path, dataSourceKeyPlainText,
          cryptoMode, minPartitions)
    }
  }

  /**
   * Interface for loading data in external storage to Dataset.
   * @param cryptoMode crypto mode, such as PLAIN_TEXT or AES_CBC_PKCS5PADDING
   * @return a EncryptedDataFrameReader
   */
  def read(cryptoMode: CryptoMode): EncryptedDataFrameReader = {
    new EncryptedDataFrameReader(sparkSession, cryptoMode, dataSourceKeyPlainText)
  }

  /**
   * Interface for loading specific data source in external storage to Datasets .
   * @param dataSource
   * @return a EncryptedDataFrameReader
   */
  def read(dataSource: DataSource): EncryptedDataFrameReader = {
    loadDataSourceKey(dataSource)
    new EncryptedDataFrameWriter(sparkSession, dataSource.encryptMode, dataSourceKeyPlainText)
  }

  /**
   * Interface for saving the content of the non-streaming Dataset out into external storage.
   * @param dataFrame dataframe to save.
   * @param cryptoMode crypto mode, such as PLAIN_TEXT or AES_CBC_PKCS5PADDING
   * @return a DataFrameWriter[Row]
   */
  def write(dataFrame: DataFrame, cryptoMode: CryptoMode): EncryptedDataFrameWriter = {
    new EncryptedDataFrameWriter(sparkSession, dataFrame, cryptoMode, dataSinkKeyPlainText)
  }

  /**
   * Interface for saving the content of the non-streaming Dataset out into specific external storage.
   * @param dataFrame dataframe to save.
   * @param dataSink
   * @return a DataFrameWriter[Row]
   */
  def write(dataFrame: DataFrame, dataSink: DataSink): EncryptedDataFrameWriter = {
    loadDataSinkKey()
    new EncryptedDataFrameWriter(sparkSession, dataFrame, dataSink.cryptoMode, dataSinkKeyPlainText)
  }

  /**
   * Get SparkSession from PPMLContext
   * @return SparkSession in PPMLContext
   */
  def getSparkSession(): SparkSession = {
    sparkSession
  }

  def getDataKeyPlainTextFromKMS(primaryKeyCertificate: String,
               dataKeyCertificate: String,
               kms: KeyManagementService): String = {
    val dataKeyPlainText = kms.retrieveDataKeyPlainText(
        primaryKeyCertificate,
        dataKeyCertificate,
        sparkSession.sparkContext.hadoopConfiguration)
    dataKeyPlainText
  }

  def getKeysFromConf(conf: SparkConf): (String, String) = {
    Log4Error.invalidInputError(conf.contains("spark.bigdl.kms.dataKey"),
        "Data key not found, please provide" +
        " both spark.bigdl.kms.primaryKey and spark.bigdl.kms.dataKey.")
      val primaryKey = conf.get("spark.bigdl.kms.primaryKey")
      val dataKey = conf.get("spark.bigdl.kms.dataKey")
      (primaryKey, dataKey)
  }

  def loadKMS(kmsType: String, conf: SparkConf): KeyManagementService = {
    val kms = kmsType match {
      case KMS_CONVENTION.MODE_EHSM_KMS =>
        val ip = conf.get("spark.bigdl.kms.ip")
        val port = conf.get("spark.bigdl.kms.port")
        val appId = conf.get("spark.bigdl.kms.appId")
        val apiKey = conf.get("spark.bigdl.kms.apiKey")
        new EHSMKeyManagementService(ip, port, appId, apiKey)
      case KMS_CONVENTION.MODE_SIMPLE_KMS =>
        val appId = conf.get("spark.bigdl.kms.appId", defaultValue = "simpleAPPID")
        val apiKey = conf.get("spark.bigdl.kms.simple.apiKey", defaultValue = "simpleAPIKEY")
        SimpleKeyManagementService(appId, apiKey)
      case KMS_CONVENTION.MODE_AZURE_KMS =>
        val vaultName = conf.get("spark.bigdl.kms.vault")
        val clientId = conf.get("spark.bigdl.kms.clientId")
        new AzureKeyManagementService(vaultName, clientId)
      case KMS_CONVENTION.MODE_BIGDL_KMS =>
        val ip = conf.get("spark.bigdl.kms.ip")
        val port = conf.get("spark.bigdl.kms.port")
        val userName = conf.get("spark.bigdl.kms.user")
        val userToken = conf.get("spark.bigdl.kms.token")
        new BigDLKeyManagementService(ip, port, userName, userToken)
      case _ =>
        throw new EncryptRuntimeException("Wrong kms type")
    }
    kms
  }

  def loadKMS(conf: SparkConf, kmsName: String): KeyManagementService = {
    Log4Error.invalidInputError(conf.contains(s"spark.bigdl.kms.$kmsName.type"),
        s"spark.bigdl.kms.$kmsName.type not found.")
    val kmsType = conf.get(s"spark.bigdl.kms.$kmsName.type")
    val kms = kmsType match {
      case KMS_CONVENTION.MODE_EHSM_KMS =>
        val ip = conf.get(s"spark.bigdl.kms.$kmsName.ip")
        val port = conf.get(s"spark.bigdl.kms.$kmsName.port")
        val appId = conf.get(s"spark.bigdl.kms.$kmsName.appId")
        val apiKey = conf.get(s"spark.bigdl.kms.$kmsName.apiKey")
        new EHSMKeyManagementService(ip, port, appId, apiKey)
      case KMS_CONVENTION.MODE_SIMPLE_KMS =>
        val appId = conf.get(s"spark.bigdl.kms.$kmsName.appId", defaultValue = "simpleAPPID")
        val apiKey = conf.get(s"spark.bigdl.kms.$kmsName.apiKey", defaultValue = "simpleAPIKEY")
        SimpleKeyManagementService(appId, apiKey)
      case KMS_CONVENTION.MODE_AZURE_KMS =>
        val vaultName = conf.get(s"spark.bigdl.kms.$kmsName.vault")
        val clientId = conf.get(s"spark.bigdl.kms.$kmsName.clientId")
        new AzureKeyManagementService(vaultName, clientId)
      case KMS_CONVENTION.MODE_BIGDL_KMS =>
        val ip = conf.get(s"spark.bigdl.kms.$kmsName.ip")
        val port = conf.get(s"spark.bigdl.kms.$kmsName.port")
        val userName = conf.get(s"spark.bigdl.kms.$kmsName.user")
        val userToken = conf.get(s"spark.bigdl.kms.$kmsName.token")
        new BigDLKeyManagementService(ip, port, userName, userToken)
      case _ =>
        throw new EncryptRuntimeException("Wrong kms type")
    }
    kms
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
        crypto.init(cryptoMode, DECRYPT, dataKeyPlaintext)
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
   * init ppml context with an existed SparkSession
   * @param sparkSession a SparkSession
   * @return a PPMLContext
   */
  def initPPMLContext(sparkSession: SparkSession): PPMLContext = {
    val conf = sparkSession.sparkContext.getConf
    val ppmlSc = loadPPMLContext(conf)
    ppmlSc
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
    conf.set("spark.hadoop.io.compression.codecs",
        "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec")
    val sc = initNNContext(conf, appName)
    val sparkSession: SparkSession = SparkSession.builder().getOrCreate()
    val ppmlSc = loadPPMLContext(conf)
    ppmlSc
  }

  /**
   * init ppml context with a sparkSession where multiple KMSs exist
   * @param sparkSession a SparkSession
   * @return a PPMLContext
   */
  def initPPMLMultiKMSContext(sparkSession: SparkSession): PPMLContext = {
    // get sparkconf and init ppml context
    val conf = sparkSession.sparkContext.getConf
    val ppmlSc = new PPMLContext(sparkSession = sparkSession)
    // init kmsManagement
    val kmsNames = getKMSNames(conf)
    kmsNames.foreach{
        kmsName => {
            val kms = loadKMS(conf, kmsName)
            ppmlSc.kmsManagement.enrollKMS(kmsName, kms)
        }
    }
    ppmlSc
  }

  def loadPPMLContext(conf: SparkConf): PPMLContext = {
    val kmsType = conf.get("spark.bigdl.kms.type", defaultValue = "SimpleKeyManagementService")
    val kms = loadKMS(kmsType, conf)
    val ppmlSc = new PPMLContext(kms, sparkSession)
    if (conf.contains("spark.bigdl.kms.primaryKey")) {
      val (primaryKey, dataKey) = getKeysFromConf(conf)
      ppmlSc.loadKeys(primaryKey, dataKey)
    }
    ppmlSc
  }

  def getKMSNames(conf: SparkConf): List[String] = {
    val names = conf.getAllWithPrefix("spark.bigdl.kms")
                    .map(_ => (_._1).substring(prefix.length + 1).split(".").first)
                    .unique()
    names
  }

}
