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
import com.intel.analytics.bigdl.ppml.crypto.CryptoMode
import com.intel.analytics.bigdl.ppml.crypto.CryptoMode.CryptoMode
import com.intel.analytics.bigdl.ppml.crypto.dataframe.EncryptedDataFrameReader.toDataFrame
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
 *
 * @param sparkSession
 * @param encryptMode
 * @param dataKeyPlainText
 */
class EncryptedDataFrameReader(
      sparkSession: SparkSession,
      encryptMode: CryptoMode,
      dataKeyPlainText: String) {
  protected val extraOptions = new scala.collection.mutable.HashMap[String, String]

  def option(key: String, value: String): this.type = {
    this.extraOptions += (key -> value)
    this
  }
  def csv(path: String): DataFrame = {
    encryptMode match {
      case CryptoMode.PLAIN_TEXT =>
        sparkSession.read.options(extraOptions).csv(path)
      case CryptoMode.AES_CBC_PKCS5PADDING =>
        val rdd = PPMLContext.textFile(sparkSession.sparkContext, path,
          dataKeyPlainText)
        // TODO: support more options
        if (extraOptions.contains("header") &&
          extraOptions("header").toLowerCase() == "true") {
          EncryptedDataFrameReader.toDataFrame(rdd)
        } else {
          val rows = rdd.map(_.split(",")).map(Row.fromSeq(_))
          val fields = (0 until  rows.first().length).map(i =>
            StructField(s"_c$i", StringType, true)
          )
          val schema = StructType(fields)
          sparkSession.createDataFrame(rows, schema)
        }
      case _ =>
        throw new IllegalArgumentException("unknown EncryptMode " + CryptoMode.toString)
    }
  }
}

object EncryptedDataFrameReader {

  /**
   * Convert RDD to DataFrame, assume the first element of RDD is the header
   * @param dataRDD The original RDD
   * @return a DataFrame
   */
  private[bigdl] def toDataFrame(dataRDD: RDD[String]): DataFrame = {
    // get schema
    val sparkSession: SparkSession = SparkSession.builder().getOrCreate()
    val header = dataRDD.first()
    val schemaArray = header.split(",")
    val fields = schemaArray.map(fieldName => StructField(fieldName, StringType, true))
    val schema = StructType(fields)

    // remove title line
    val data_filter = dataRDD.filter(row => row != header)

    // create df
    val rowRdd = data_filter.map(s => Row.fromSeq(s.split(",")))
    sparkSession.createDataFrame(rowRdd, schema)
  }
}
