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

import com.intel.analytics.bigdl.ppml.crypto.dataframe.EncryptedDataFrameWriter.writeCsv
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, Crypto, CryptoMode, ENCRYPT, PLAIN_TEXT}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SerializableWritable, SparkContext, TaskContext}
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

import java.util.Locale

class EncryptedDataFrameWriter(
      sparkSession: SparkSession,
      df: DataFrame,
      encryptMode: CryptoMode,
      dataKeyPlainText: String) {
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
      case _ => throw new IllegalArgumentException(s"Unknown save mode: $saveMode. " +
        "Accepted save modes are 'overwrite', 'append', 'ignore', 'error', 'errorifexists'.")
    }
    this
  }

  def csv(path: String): Unit = {
    encryptMode match {
      case PLAIN_TEXT =>
        df.write.options(extraOptions).csv(path)
      case AES_CBC_PKCS5PADDING =>
        writeCsv(df.rdd, sparkSession.sparkContext, path, encryptMode, dataKeyPlainText)
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
                         dataKeyPlainText: String): Unit = {
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
        val cypto = Crypto(cryptoMode = encryptMode)
        cypto.init(encryptMode, ENCRYPT, dataKeyPlainText)
        val header = cypto.genHeader()

        output.write(header)
        var row = rows.next()
        while (rows.hasNext) {
          val line = row.toSeq.mkString(",") + "\n"
          output.write(cypto.update(line.getBytes))
//          print(rows.hasNext)
          row = rows.next()
        }
        val lastLine = row.toSeq.mkString(",")
        val (lBytes, hmac) = cypto.doFinal(lastLine.getBytes)
        output.write(lBytes)
        output.write(hmac)
        output.flush()
        output.close()
      }
      Iterator.single(1)
    }}.count()

  }
}
