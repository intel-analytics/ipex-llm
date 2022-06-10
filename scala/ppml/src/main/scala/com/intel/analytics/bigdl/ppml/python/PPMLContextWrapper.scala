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
import com.intel.analytics.bigdl.ppml.crypto.{CryptoMode, EncryptRuntimeException}
import com.intel.analytics.bigdl.ppml.crypto.dataframe.{EncryptedDataFrameReader, EncryptedDataFrameWriter}
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

  def write(sc: PPMLContext, dataFrame: DataFrame,
            cryptoModeStr: String): EncryptedDataFrameWriter = {
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

}
