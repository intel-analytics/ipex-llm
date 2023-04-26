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

import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.crypto.dataframe.EncryptedDataFrameWriter.writeCsv
import com.intel.analytics.bigdl.ppml.utils.KeyReaderWriter
import com.intel.analytics.bigdl.ppml.kms.common.KeyLoaderManagement
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, AES_GCM_CTR_V1, AES_GCM_V1, Crypto, CryptoMode, ENCRYPT, PLAIN_TEXT}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SerializableWritable, SparkContext, TaskContext}
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}


import java.nio.charset.StandardCharsets
import java.util.{Base64, Locale}

class EncryptedDataFrameWriter(
      sparkSession: SparkSession,
      df: DataFrame,
      encryptMode: CryptoMode,
      primaryKeyName: String,
      keyLoaderManagement: KeyLoaderManagement) {
  protected val extraOptions = new scala.collection.mutable.HashMap[String, String]

  def option(key: String, value: String): this.type = {
    this.extraOptions += (key -> value)
    this
  }

  def option(key: String, value: Boolean): this.type = {
    this.extraOptions += (key -> value.toString)
    this
  }

  private var mode: SaveMode = SaveMode.ErrorIfExists

  def mode(saveMode: String): this.type = {
    this.mode = saveMode.toLowerCase(Locale.ROOT) match {
      case "overwrite" => SaveMode.Overwrite
      case "append" => SaveMode.Append
      case "ignore" => SaveMode.Ignore
      case "error" | "errorifexists" | "default" => SaveMode.ErrorIfExists
      case _ =>
        Log4Error.invalidOperationError(false,
          s"Unknown save mode: $saveMode. " +
        "Accepted save modes are 'overwrite', 'append', 'ignore', 'error', 'errorifexists'.")
        null
    }
    this
  }

  def setCryptoCodecContext(): Unit = {
    encryptMode match {
      case PLAIN_TEXT =>
      case AES_CBC_PKCS5PADDING =>
        val (dataKeyPlainText, dataKeyCipherText) = keyLoaderManagement
          .retrieveKeyLoader(primaryKeyName).generateDataKeyPlainText
        sparkSession.sparkContext.hadoopConfiguration
                    .set("bigdl.write.dataKey.plainText", dataKeyPlainText)
        if (dataKeyCipherText != null) {
          // native AES CBC does not need dataKeyCipherText
          sparkSession.sparkContext.hadoopConfiguration
            .set("bigdl.write.dataKey.cipherText", dataKeyCipherText)
        }
        sparkSession.sparkContext.hadoopConfiguration
                    .set("bigdl.cryptoMode", encryptMode.encryptionAlgorithm)
        option("compression", "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec")
      case _ =>
        Log4Error.invalidOperationError(false,
          "unknown or wrong encryptMode " + CryptoMode.toString)
    }
  }

  def writeEncryptedDataKeyToMeta(path: String): Unit = {
    encryptMode match {
      case PLAIN_TEXT =>
      case AES_CBC_PKCS5PADDING =>
        keyLoaderManagement
          .retrieveKeyLoader(primaryKeyName)
          .writeEncryptedDataKey(path)
      case _ =>
        Log4Error.invalidOperationError(false,
          "unknown or wrong encryptMode " + CryptoMode.toString)
    }
  }

  def csv(path: String): Unit = {
    setCryptoCodecContext()
    df.write.options(extraOptions).mode(mode).csv(path)
    writeEncryptedDataKeyToMeta(path)
  }

  def json(path: String): Unit = {
    setCryptoCodecContext()
    df.write.options(extraOptions).mode(mode).json(path)
    writeEncryptedDataKeyToMeta(path)
  }

  def text(path: String): Unit = {
    setCryptoCodecContext()
    df.write.options(extraOptions).mode(mode).text(path)
    writeEncryptedDataKeyToMeta(path)
  }

  def parquet(path: String): Unit = {
    lazy val header = df.schema.fieldNames.mkString(",")
    encryptMode match {
      case PLAIN_TEXT =>
        df.write.options(extraOptions).mode(mode).parquet(path)
      case AES_GCM_CTR_V1 | AES_GCM_V1 =>
        val dataKeyPlainText = keyLoaderManagement.retrieveKeyLoader(primaryKeyName)
                                                  .generateDataKeyPlainText._1
        EncryptedDataFrameWriter.setParquetKey(sparkSession, dataKeyPlainText)
        df.write
          .option("parquet.encryption.column.keys", "key1: " + header)
          .option("parquet.encryption.footer.key", "footerKey")
          .option("parquet.encryption.algorithm", encryptMode.encryptionAlgorithm)
          .options(extraOptions).mode(mode).parquet(path)
        keyLoaderManagement
          .retrieveKeyLoader(primaryKeyName)
          .writeEncryptedDataKey(path)
      case _ =>
        throw new IllegalArgumentException("unknown EncryptMode " + CryptoMode.toString)
    }
  }

}

object EncryptedDataFrameWriter {
  protected def writeCsv(rdd: RDD[Row],
                         sc: SparkContext,
                         path: String,
                         encryptMode: CryptoMode,
                         dataKeyPlainText: String,
                         schema: String): Unit = {
    val confBroadcast = sc.broadcast(
      new SerializableWritable(sc.hadoopConfiguration)
    )
    rdd.mapPartitions{ rows => {
      if (rows.nonEmpty) {
        val hadoopConf = confBroadcast.value.value
        val fs = FileSystem.get(hadoopConf)
        val partId = TaskContext.getPartitionId()
        // TODO
        val output = fs.create(new Path(path + "/part-" + partId), true)
        val crypto = Crypto(cryptoMode = encryptMode)
        crypto.init(encryptMode, ENCRYPT, dataKeyPlainText)
        // write crypto header
        output.write(crypto.genHeader())
        // write csv header
        if (schema != null && schema.nonEmpty) {
          output.write(crypto.update((schema + "\n").getBytes))
        }
        var row = rows.next()
        while (rows.hasNext) {
          val line = row.toSeq.mkString(",") + "\n"
          output.write(crypto.update(line.getBytes))
//          print(rows.hasNext)
          row = rows.next()
        }
        val lastLine = row.toSeq.mkString(",")
        val (lBytes, hmac) = crypto.doFinal(lastLine.getBytes)
        output.write(lBytes)
        output.write(hmac)
        output.flush()
        output.close()
      }
      Iterator.single(1)
    }}.count()

  }

  private[bigdl] def setParquetKey(sparkSession: SparkSession, dataKeyPlainText: String): Unit = {
    val encoder = Base64.getEncoder
    val footKey = encoder.encodeToString(dataKeyPlainText.slice(0, 16)
      .getBytes(StandardCharsets.UTF_8))
    val dataKey = encoder.encodeToString(dataKeyPlainText.slice(16, 32)
      .getBytes(StandardCharsets.UTF_8))
    sparkSession.sparkContext.hadoopConfiguration.set("parquet.crypto.factory.class",
      "org.apache.parquet.crypto.keytools.PropertiesDrivenCryptoFactory")
    sparkSession.sparkContext.hadoopConfiguration.set("parquet.encryption.kms.client.class",
      "org.apache.parquet.crypto.keytools.mocks.InMemoryKMS")
    sparkSession.sparkContext.hadoopConfiguration.set("parquet.encryption.key.list",
      s"footerKey: ${footKey}, key1: ${dataKey}")
  }
}
