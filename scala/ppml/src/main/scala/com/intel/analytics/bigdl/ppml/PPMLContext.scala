package com.intel.analytics.bigdl.ppml

import com.intel.analytics.bigdl.dllib.NNContext.{checkScalaVersion, checkSparkVersion, createSparkConf, initConf, initNNContext}
import com.intel.analytics.bigdl.ppml.encrypt.EncryptMode.EncryptMode
import com.intel.analytics.bigdl.ppml.encrypt.{EncryptMode, EncryptRuntimeException, FernetEncrypt}
import com.intel.analytics.bigdl.ppml.utils.Supportive
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.input.PortableDataStream
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, DataFrameReader, DataFrameWriter, Row, SparkSession}
import com.intel.analytics.bigdl.ppml.kms.{EHSMKeyManagementService, KMS_CONVENTION, KeyManagementService, SimpleKeyManagementService}
import com.intel.analytics.bigdl.ppml.dataframe.EncryptedDataFrameReader

import java.nio.file.Paths

class PPMLContext(kms: KeyManagementService, sparkSession: SparkSession) {

  private var dataKeyPlainText: String = ""

  def loadKeys(primaryKeyPath: String, dataKeyPath: String): this.type = {
    dataKeyPlainText = kms.retrieveDataKeyPlainText(
      Paths.get(primaryKeyPath).toString, Paths.get(dataKeyPath).toString)
    this
  }

  def textFile(path: String,
               minPartitions: Int = sparkSession.sparkContext.defaultMinPartitions,
               mode: EncryptMode = EncryptMode.PLAIN_TEXT): RDD[String] = {
    mode match {
      case EncryptMode.PLAIN_TEXT =>
        sparkSession.sparkContext.textFile(path, minPartitions)
      case EncryptMode.AES_CBC_PKCS5PADDING =>
        PPMLContext.textFile(sparkSession.sparkContext, path, dataKeyPlainText, minPartitions)
      case _ =>
        throw new IllegalArgumentException("unknown EncryptMode " + mode.toString)
    }
  }

  def read(mode: EncryptMode): EncryptedDataFrameReader = {
    new EncryptedDataFrameReader(sparkSession, mode, dataKeyPlainText)
  }

  def write(dataFrame: DataFrame, mode: EncryptMode): DataFrameWriter[Row] = {
    mode match {
      case EncryptMode.PLAIN_TEXT =>
        dataFrame.write
      case EncryptMode.AES_CBC_PKCS5PADDING =>
        PPMLContext.write(sparkSession, dataKeyPlainText, dataFrame)
      case _ =>
        throw new IllegalArgumentException("unknown EncryptMode " + mode.toString)
    }
  }
}

object PPMLContext{
  def apply(): PPMLContext = {
    val sparkSession: SparkSession = SparkSession.builder().getOrCreate()
    apply(sparkSession)
  }

  def apply(sparkSession: SparkSession): PPMLContext = {
    val skms = new SimpleKeyManagementService()
    apply(skms, sparkSession)
  }

  def apply(kms: KeyManagementService): PPMLContext = {
    val sparkSession: SparkSession = SparkSession.builder().getOrCreate()
    new PPMLContext(kms, sparkSession)
  }

  def apply(kms: KeyManagementService, sparkSession: SparkSession): PPMLContext = {
    new PPMLContext(kms, sparkSession)
  }


  def registerUDF(spark: SparkSession,
                  dataKeyPlaintext: String) = {
    val bcKey = spark.sparkContext.broadcast(dataKeyPlaintext)
    val convertCase = (x: String) => {
      val fernetCryptos = new FernetEncrypt()
      new String(fernetCryptos.encryptBytes(x.getBytes, bcKey.value))
    }
    spark.udf.register("convertUDF", convertCase)
  }

  def textFile(sc: SparkContext,
               path: String,
               dataKeyPlaintext: String,
               minPartitions: Int = -1): RDD[String] = {
    require(dataKeyPlaintext != "", "dataKeyPlainText should not be empty, please loadKeys first.")
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

  def write(sparkSession: SparkSession,
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

  def writeCsv(sparkSession: SparkSession,
               dataKeyPlaintext: String,
               dataFrame: DataFrame,
               path: String): Unit = {
    val tableName = "save"
    dataFrame.createOrReplaceTempView(tableName)
    PPMLContext.registerUDF(sparkSession, dataKeyPlaintext)
    // Select all and encrypt columns.
    val convertSql = "select " + dataFrame.schema.map(column =>
      "convertUDF(" + column.name + ") as " + column.name).mkString(", ") +
      " from " + tableName
    val df = sparkSession.sql(convertSql)
    df.write.mode("overwrite").option("header", true).csv(Paths.get(path, tableName).toString)
  }

  def initPPMLContext(appName: String): PPMLContext = {
    initPPMLContext(null, appName)
  }

  def initPPMLContext(conf: SparkConf): PPMLContext = {
    initPPMLContext(conf)
  }

  def initPPMLContext(
        appName: String,
        ppmlArgs: Map[String, String]): PPMLContext = {
    initPPMLContext(null, appName, ppmlArgs)
  }

  def initPPMLContext(
        sparkConf: SparkConf,
        appName: String,
        ppmlArgs: Map[String, String]): PPMLContext = {
    val conf = createSparkConf(sparkConf)
    val sc = initNNContext(conf, appName)
    val sparkSession: SparkSession = SparkSession.builder().getOrCreate()
    val kmsType = ppmlArgs.get("spark.bigdl.kms.type").get
    val kms = kmsType match {
      case KMS_CONVENTION.MODE_EHSM_KMS =>
        val ip = ppmlArgs.getOrElse("spark.bigdl.kms.ehs.ip", "0.0.0.0")
        val port = ppmlArgs.getOrElse("spark.bigdl.kms.ehs.port", "5984")
        val appId = ppmlArgs.getOrElse("spark.bigdl.kms.ehs.id", "ehsmAPPID")
        val appKey = ppmlArgs.getOrElse("spark.bigdl.kms.ehs.key", "ehsmAPPKEY")
        new EHSMKeyManagementService(ip, port, appId, appKey)
      case KMS_CONVENTION.MODE_SIMPLE_KMS =>
        val id = ppmlArgs.getOrElse("spark.bigdl.kms.simple.id", "simpleAPPID")
        val key = ppmlArgs.getOrElse("spark.bigdl.kms.simple.key", "simpleAPPKEY")
        new SimpleKeyManagementService(id, key)
      case _ =>
        throw new EncryptRuntimeException("Wrong kms type")
    }
    val kmsSc = new PPMLContext(kms, sparkSession)
    if (ppmlArgs.contains("spark.bigdl.kms.key.primary")){
      require(ppmlArgs.contains("spark.bigdl.kms.key.data"), "Data key not found, please provide" +
        " both spark.bigdl.kms.key.primary and spark.bigdl.kms.key.data.")
      val primaryKey = ppmlArgs.get("spark.bigdl.kms.key.primary").get
      val dataKey = ppmlArgs.get("spark.bigdl.kms.key.data").get
      kmsSc.loadKeys(primaryKey, dataKey)
    }
    kmsSc
  }

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
    val kmsSc = new PPMLContext(kms, sparkSession)
    if (conf.contains("spark.bigdl.kms.key.primary")){
      require(conf.contains("spark.bigdl.kms.key.data"), "Data key not found, please provide" +
        " both spark.bigdl.kms.key.primary and spark.bigdl.kms.key.data.")
      val primaryKey = conf.get("spark.bigdl.kms.key.primary")
      val dataKey = conf.get("spark.bigdl.kms.key.data")
      kmsSc.loadKeys(primaryKey, dataKey)
    }
    kmsSc
  }

}
