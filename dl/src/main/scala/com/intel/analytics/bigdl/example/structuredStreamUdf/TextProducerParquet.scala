/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.example.structuredStreamUdf

import java.io.File
import com.intel.analytics.bigdl.example.structuredStreamUdf.Options._
import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession

import scala.io.Source

object TextProducerParquet {
  val logger = Logger.getLogger(getClass)

  case class Sample(filename: String, text: String)

  def loadTestData(testDir: String): Array[Sample] = {
    val fileList = new File(testDir).listFiles()
      .filter(_.isFile).filter(_.getName.forall(Character.isDigit(_))).sorted

    val testData = fileList.map { file => {
      val fileName = file.getName
      val source = Source.fromFile(file, "ISO-8859-1")
      val text = try source.getLines().toList.mkString("\n") finally source.close()
      new Sample(fileName, text)
    }
    }
    testData
  }

  def main(args: Array[String]): Unit = {

    parquetProducerParser.parse(args, TextProducerParquetParams()).map { param =>

      val batchsize = param.batchsize
      val interval = param.interval
      // load messages
      val data = loadTestData(param.srcFolder)

      val spark = SparkSession.builder().appName("Produce Text").getOrCreate()
      val batch: Array[Sample] = new Array[Sample](batchsize)
      var count = 0
      var send_count = 0
      val iter = data.iterator
      while (iter.hasNext) {
        try {
          if (count < batchsize) {
            batch(count) = iter.next()
            count += 1
          } else if (count == batchsize) {
            // send batch
            val testRDD = spark.sparkContext.parallelize(batch, 1)
            val df = spark.createDataFrame(testRDD)
            println("send text batch " + send_count)
            df.write
              .format("parquet")
              .mode(org.apache.spark.sql.SaveMode.Append)
              .save(param.destFolder)
            count = 0
            send_count += 1
            Thread.sleep(interval*1000)
          }

        } catch {
          case e: Exception => println(e)
        }
      }
    }
  }
}
