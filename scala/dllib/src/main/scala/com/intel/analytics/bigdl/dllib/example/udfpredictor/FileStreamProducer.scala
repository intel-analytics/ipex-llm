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
package com.intel.analytics.bigdl.example.udfpredictor

import com.intel.analytics.bigdl.example.udfpredictor.Utils._
import com.intel.analytics.bigdl.utils.LoggerFilter
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}
import org.slf4j.{Logger, LoggerFactory}


object FileStreamProducer {

  val log: Logger = LoggerFactory.getLogger(this.getClass)
  LoggerFilter.redirectSparkInfoLogs()
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Levle4j.INFO)

  def main(args: Array[String]): Unit = {

    parquetProducerParser.parse(args, TextProducerParquetParams()).foreach { param =>

      val batchSize = param.batchsize
      val interval = param.interval
      // load messages
      val data = Utils.loadTestData(param.srcFolder)

      val spark = SparkSession.builder().appName("Produce Text").getOrCreate()
      var send_count = 0
      val batches = data.grouped(batchSize)
      batches.foreach { batch =>
        try {
          val df = spark.createDataFrame(batch)
          log.info("send text batch " + send_count)
          df.write
            .format("parquet")
            .mode(org.apache.spark.sql.SaveMode.Append)
            .save(param.destFolder)
          send_count += 1
          Thread.sleep(interval*1000)
        } catch {
          case e: Exception => log.error("sending batch error", e)
        }
      }
    }
  }
}
