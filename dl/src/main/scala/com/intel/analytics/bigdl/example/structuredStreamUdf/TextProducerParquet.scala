
package com.intel.analytics.bigdl.example.structuredStreamUdf

import java.io.File
import com.intel.analytics.bigdl.example.structuredStreamUdf.Options._
import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession

import scala.io.Source

/**
  * Created by jwang on 2/21/17.
  */
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
      var iter = data.iterator
      val spark = SparkSession.builder().appName("Produce Text").getOrCreate()
      val batch: Array[Sample] = new Array[Sample](batchsize)
      var count = 0
      var send_count = 0
      while (iter.hasNext) {
        try {
          if (count < batchsize) {
            batch(count) = iter.next()
            count += 1
          } else if (count == batchsize) {
            // send
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
