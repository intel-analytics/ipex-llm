package com.intel.analytics.bigdl.ppml.python

import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.{CryptoMode, EncryptRuntimeException}
import com.intel.analytics.bigdl.ppml.crypto.dataframe.EncryptedDataFrameReader
import com.intel.analytics.bigdl.ppml.kms.KMS_CONVENTION
import org.apache.spark.sql.{DataFrame, DataFrameWriter, Row}
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import java.util

object PPMLContextWrapper {
  def ofFloat: PPMLContextWrapper[Float] = new PPMLContextWrapper[Float]()

  def ofDouble: PPMLContextWrapper[Double] = new PPMLContextWrapper[Double]()
}

class PPMLContextWrapper[T]() {
  val logger: Logger = LoggerFactory.getLogger(getClass)

  def createPPMLContext(appName: String): PPMLContext = {
    logger.debug("create PPMLContextWrapper with appName")
    PPMLContext.initPPMLContext(appName)
  }

  def createPPMLContext(appName: String, ppmlArgs: util.Map[String, String]): PPMLContext = {
    logger.debug("create PPMLContextWrapper with appName & ppmlArgs")

    val kmsArgs = scala.collection.mutable.Map[String, String]()
    val kmsType = ppmlArgs.get("kms_type")
    kmsArgs("spark.bigdl.kms.type") = kmsType
    kmsType match {
      case KMS_CONVENTION.MODE_EHSM_KMS =>
        throw new EncryptRuntimeException("not support yet")
      case KMS_CONVENTION.MODE_SIMPLE_KMS =>
        kmsArgs("spark.bigdl.kms.simple.id") = ppmlArgs.get("simple_app_id")
        kmsArgs("spark.bigdl.kms.simple.key") = ppmlArgs.get("simple_app_key")
      case _ =>
        throw new EncryptRuntimeException("Wrong kms type")
    }
    if (new File(ppmlArgs.get("primary_key_path")).exists()) {
      kmsArgs("spark.bigdl.kms.key.primary") = ppmlArgs.get("primary_key_path")
    }
    if (new File(ppmlArgs.get("data_key_path")).exists()) {
      kmsArgs("spark.bigdl.kms.key.data") = ppmlArgs.get("data_key_path")
    }
    PPMLContext.initPPMLContext(appName, kmsArgs.toMap)
  }

  def read(sc: PPMLContext, cryptoModeStr: String): EncryptedDataFrameReader = {
    logger.debug("read file with crypto mode " + cryptoModeStr)
    val cryptoMode = CryptoMode.parse(cryptoModeStr)
    sc.read(cryptoMode)
  }

  def write(sc: PPMLContext, dataFrame: DataFrame, cryptoModeStr: String): DataFrameWriter[Row] = {
    logger.debug("write file with crypt mode " + cryptoModeStr)
    val cryptoMode = CryptoMode.parse(cryptoModeStr)
    sc.write(dataFrame, cryptoMode)
  }

  def loadKeys(sc: PPMLContext,
               primaryKeyPath: String, dataKeyPath: String): Unit = {
    sc.loadKeys(primaryKeyPath, dataKeyPath)
  }

  /**
   * EncryptedDataFrameReader method
   */

  def option(encryptedDataFrameReader: EncryptedDataFrameReader,
             key: String, value: String): EncryptedDataFrameReader = {
    encryptedDataFrameReader.option(key, value)
  }

  def csv(encryptedDataFrameReader: EncryptedDataFrameReader, path: String): DataFrame = {
    logger.debug("read csv file from path: " + path)
    encryptedDataFrameReader.csv(path)
  }
}
