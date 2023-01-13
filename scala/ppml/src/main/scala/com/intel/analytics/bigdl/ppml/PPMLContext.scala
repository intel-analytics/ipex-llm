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
 * @param kms
 * @param sparkSession
 */
class PPMLContext protected(kms: KeyManagementService = null, sparkSession: SparkSession = null) {

  protected var dataKeyPlainText: String = ""
  protected val multiKms = new scala.collection.mutable.HashMap[String, KeyManagementService]
  protected val dataSources = new scala.collection.mutable.HashMap[String, String]

  def addKMS(kmsName: String, kms: KeyManagementService): this.type = {
    this.multiKms +=  (kmsName -> kms)
    this
  }

  def addDataSource(dataSourceName: String, kmsName: String): this.type = {
    this.dataSources += (dataSourceName->kmsName)
    this
  }

  // def getKMS(dataSourceName:String): KeyManagementService{
  //   this.multiKms.get(this.dataSources.get(dataSourceName))
  // }

  /**
   * Load keys from a local file system.
   * @param primaryKeyPath
   * @param dataKeyPath
   * @return
   */

  def loadKeys(primaryKeyPath: String, dataKeyPath: String): this.type = {
    dataKeyPlainText = kms.retrieveDataKeyPlainText(
      new Path(primaryKeyPath).toString, new Path(dataKeyPath).toString,
      sparkSession.sparkContext.hadoopConfiguration)
    sparkSession.sparkContext.hadoopConfiguration.set("bigdl.kms.data.key", dataKeyPlainText)
    this
  }

  // specified one datasource to loadkeys
  def loadKeys(primaryKeyPath: String,
               dataKeyPath: String,
               dataSourceName: String ): this.type = {
    Log4Error.invalidInputError(this.dataSources.contains(dataSourceName),
      "this.dataSources.get(dataSourceName).get==null, cant get kms of " + dataSourceName)
    val kmsName = this.dataSources.get(dataSourceName).get
    Log4Error.invalidInputError(this.multiKms.contains(kmsName),
      " get kms == null, cant get kms of KMSname " + kmsName)
    val kms = this.multiKms.get(kmsName).get
    Log4Error.invalidInputError(kms != null, "LOAD KEYS: kms not found")
    dataKeyPlainText = kms.retrieveDataKeyPlainText(
      new Path(primaryKeyPath).toString, new Path(dataKeyPath).toString,
      sparkSession.sparkContext.hadoopConfiguration)
    sparkSession.sparkContext.hadoopConfiguration.set("bigdl.kms." + dataSourceName + ".key",
      dataKeyPlainText)
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
        PPMLContext.textFile(sparkSession.sparkContext, path, dataKeyPlainText,
          cryptoMode, minPartitions)
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
  def write(dataFrame: DataFrame, cryptoMode: CryptoMode): EncryptedDataFrameWriter = {
    new EncryptedDataFrameWriter(sparkSession, dataFrame, cryptoMode, dataKeyPlainText)
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
        crypto.init(cryptoMode, DECRYPT, dataKeyPlaintext)
        crypto.decryptBigContent(inputDataStream)
      }
    }} // .flatMap(_.split("\n")).flatMap(_.split("\r"))
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
    val kmsType = conf.get("spark.bigdl.kms.type", defaultValue = "SimpleKeyManagementService")
    val kms = kmsType match {
      case KMS_CONVENTION.MODE_EHSM_KMS =>
        val ip = conf.get("spark.bigdl.kms.ehs.ip")
        val port = conf.get("spark.bigdl.kms.ehs.port")
        val appId = conf.get("spark.bigdl.kms.ehs.id")
        val apiKey = conf.get("spark.bigdl.kms.ehs.key")
        new EHSMKeyManagementService(ip, port, appId, apiKey)
      case KMS_CONVENTION.MODE_SIMPLE_KMS =>
        val id = conf.get("spark.bigdl.kms.simple.id", defaultValue = "simpleAPPID")
        // println(id + "=-------------------")
        val key = conf.get("spark.bigdl.kms.simple.key", defaultValue = "simpleAPIKEY")
        // println(key + "=-------------------")
        SimpleKeyManagementService(id, key)
      case KMS_CONVENTION.MODE_AZURE_KMS =>
        val vaultName = conf.get("spark.bigdl.kms.azure.vault")
        val clientId = conf.get("spark.bigdl.kms.azure.clientId")
        new AzureKeyManagementService(vaultName, clientId)
      case KMS_CONVENTION.MODE_BIGDL_KMS =>
        val ip = conf.get("spark.bigdl.kms.bigdl.ip")
        val port = conf.get("spark.bigdl.kms.bigdl.port")
        val userName = conf.get("spark.bigdl.kms.bigdl.user")
        val userToken = conf.get("spark.bigdl.kms.bigdl.token")
        new BigDLKeyManagementService(ip, port, userName, userToken)
      case _ =>
        throw new EncryptRuntimeException("Wrong kms type")
    }
    val ppmlSc = new PPMLContext(kms, sparkSession)
    if (conf.contains("spark.bigdl.kms.key.primary")) {
      Log4Error.invalidInputError(conf.contains("spark.bigdl.kms.key.data"),
        "Data key not found, please provide" +
          " both spark.bigdl.kms.key.primary and spark.bigdl.kms.key.data.")
      val primaryKey = conf.get("spark.bigdl.kms.key.primary")
      val dataKey = conf.get("spark.bigdl.kms.key.data")
      ppmlSc.loadKeys(primaryKey, dataKey)
    }
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
    val kmsType = conf.get("spark.bigdl.kms.type", defaultValue = "SimpleKeyManagementService")
    val kms = kmsType match {
      case KMS_CONVENTION.MODE_EHSM_KMS =>
        val ip = conf.get("spark.bigdl.kms.ehs.ip", defaultValue = "0.0.0.0")
        val port = conf.get("spark.bigdl.kms.ehs.port", defaultValue = "5984")
        val appId = conf.get("spark.bigdl.kms.ehs.id", defaultValue = "ehsmAPPID")
        val apiKey = conf.get("spark.bigdl.kms.ehs.key", defaultValue = "ehsmAPIKEY")
        new EHSMKeyManagementService(ip, port, appId, apiKey)
      case KMS_CONVENTION.MODE_SIMPLE_KMS =>
        val id = conf.get("spark.bigdl.kms.simple.id", defaultValue = "simpleAPPID")
        // println(id + "=-------------------")
        val key = conf.get("spark.bigdl.kms.simple.key", defaultValue = "simpleAPIKEY")
        // println(key + "=-------------------")
        SimpleKeyManagementService(id, key)
      case KMS_CONVENTION.MODE_AZURE_KMS =>
        val vaultName = conf.get("spark.bigdl.kms.azure.vault", defaultValue = "keyVaultName")
        val clientId = conf.get("spark.bigdl.kms.azure.clientId", defaultValue = "")
        new AzureKeyManagementService(vaultName, clientId)
      case KMS_CONVENTION.MODE_BIGDL_KMS =>
        val ip = conf.get("spark.bigdl.kms.bigdl.ip")
        val port = conf.get("spark.bigdl.kms.bigdl.port")
        val userName = conf.get("spark.bigdl.kms.bigdl.user")
        val userToken = conf.get("spark.bigdl.kms.bigdl.token")
        new BigDLKeyManagementService(ip, port, userName, userToken)
      case _ =>
        throw new EncryptRuntimeException("Wrong kms type")
    }
    val ppmlSc = new PPMLContext(kms, sparkSession)
    if (conf.contains("spark.bigdl.kms.key.primary")) {
      Log4Error.invalidInputError(conf.contains("spark.bigdl.kms.key.data"),
        "Data key not found, please provide" +
        " both spark.bigdl.kms.key.primary and spark.bigdl.kms.key.data.")
      val primaryKey = conf.get("spark.bigdl.kms.key.primary")
      val dataKey = conf.get("spark.bigdl.kms.key.data")
      ppmlSc.loadKeys(primaryKey, dataKey)
    }
    ppmlSc
  }

  // init ppmlContext with multi kms and multi datasource
  def initPPMLContextMultiKMS(sparkSession: SparkSession): PPMLContext = {
    // get sparkconf and init ppml context
    val conf = sparkSession.sparkContext.getConf
    val ppmlSc = new PPMLContext(sparkSession = sparkSession)

    // init kms
    Log4Error.invalidInputError(conf.contains(s"spark.bigdl.kms.multikms.instance"),
      "spark.bigdl.kms.multikms.instance not found, please provide instance of your KMS")
    val instance = conf.getInt(s"spark.bigdl.kms.multikms.instance", defaultValue = 2)
    for (i <- 1 to instance) {
      Log4Error.invalidInputError(conf.contains(s"spark.bigdl.kms.multikms.type${i}"),
        s"spark.bigdl.kms.multikms.type${i} not found.")
      Log4Error.invalidInputError(conf.contains(s"spark.bigdl.kms.multikms.name${i}"),
        s"spark.bigdl.kms.multikms.name${i} not found.")
      val kmsType = conf.get(s"spark.bigdl.kms.multikms.type${i}")
      val kmsName = conf.get(s"spark.bigdl.kms.multikms.name${i}")
      val kms = kmsType match {
      case KMS_CONVENTION.MODE_EHSM_KMS =>
        val ip = conf.get(s"spark.bigdl.kms.multikms.ehs.ip${i}",
          defaultValue = "0.0.0.0")
        val port = conf.get(s"spark.bigdl.kms.multikms.ehs.port${i}",
          defaultValue = "5984")
        val appId = conf.get(s"spark.bigdl.kms.multikms.ehs.id${i}",
          defaultValue = "ehsmAPPID")
        val apiKey = conf.get(s"spark.bigdl.kms.multikms.ehs.key${i}",
          defaultValue = "ehsmAPIKEY")
        new EHSMKeyManagementService(ip, port, appId, apiKey)
      case KMS_CONVENTION.MODE_SIMPLE_KMS =>
        val id = conf.get(s"spark.bigdl.kms.multikms.simple.id${i}",
          defaultValue = "simpleAPPID")
        // println(id + "=-------------------")
        val key = conf.get(s"spark.bigdl.kms.multikms.simple.key${i}",
          defaultValue = "simpleAPIKEY")
        // println(key + "=-------------------")
        SimpleKeyManagementService(id, key)
      case KMS_CONVENTION.MODE_AZURE_KMS =>
        val vaultName = conf.get(s"spark.bigdl.kms.multikms.azure.vault${i}",
          defaultValue = "keyVaultName")
        val clientId = conf.get(s"spark.bigdl.kms.multikms.azure.clientId${i}",
          defaultValue = "")
        new AzureKeyManagementService(vaultName, clientId)
      case _ =>
        throw new EncryptRuntimeException("Wrong kms type")
     }
     ppmlSc.addKMS(kmsName, kms)
    }

    // init data sources
    Log4Error.invalidInputError(conf.contains("spark.bigdl.kms.datasource.instance"),
      "spark.bigdl.kms.datasource.instance not found" +
      "please provide the instance of your data sources.")
    val dataSourceInstance = conf.getInt("spark.bigdl.kms.datasource.instance", defaultValue = 2)
    for (i <- 1 to dataSourceInstance) {
      val dataSourceName = conf.get(s"spark.bigdl.kms.datasource${i}.name",
        defaultValue = s"datasource${i}")
      Log4Error.invalidInputError(conf.contains(s"spark.bigdl.kms.datasource${i}.kms"),
        s"spark.bigdl.kms.datasource${i}.kms not found")
      val kms = conf.get(s"spark.bigdl.kms.datasource${i}.kms")

      // get input and output path
      Log4Error.invalidInputError(conf.contains(s"spark.bigdl.kms.datasource${i}.inputpath"),
      s"input path of data source${i} not found, please provide input path")
      // Log4Error.invalidInputError(conf.contains(s"spark.bigdl.kms.datasource${i}.outputpath"),
      // s"output path of data source${i} not found, please provide output path")
      val inputPath = conf.get(s"spark.bigdl.kms.datasource${i}.inputpath")
      if (conf.contains(s"spark.bigdl.kms.datasource${i}.outputpath")) {
        val outputPath = conf.get(s"spark.bigdl.kms.datasource${i}.outputpath")
      } else {
        val outputPath = inputPath + ".output"
      }

      // get primary key and data key
      Log4Error.invalidInputError(conf.contains(s"spark.bigdl.kms.datasource${i}.primary"),
       s"Primary key${i} not found, please provide" +
       s" both spark.bigdl.kms.datasource${i}.primary and spark.bigdl.kms.datasource${i}.data.")
      Log4Error.invalidInputError(conf.contains(s"spark.bigdl.kms.datasource${i}.data"),
       s"Data key${i} not found, please provide" +
       s" both spark.bigdl.kms.datasource${i}.primary and spark.bigdl.kms.datasource${i}.data.")
      val primaryKey = conf.get(s"spark.bigdl.kms.datasource${i}.primary")
      val dataKey = conf.get(s"spark.bigdl.kms.datasource${i}.data")
      ppmlSc.addDataSource(dataSourceName, kms)
      ppmlSc.loadKeys(primaryKey, dataKey, dataSourceName)
    }
  ppmlSc
  }

}
