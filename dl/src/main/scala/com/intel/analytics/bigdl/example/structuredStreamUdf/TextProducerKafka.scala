
package com.intel.analytics.bigdl.example.structuredStreamUdf

import java.io.File
import java.util.Properties

import org.apache.log4j.Logger

import scala.io.Source
import kafka.producer.{KeyedMessage, Producer, ProducerConfig}
import kafka.serializer.StringEncoder
import com.intel.analytics.bigdl.example.structuredStreamUdf.Options._

/**
  * Created by jwang on 2/21/17.
  */
object TextProducerKafka {
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
//      (fileName, text)
    }
    }
    testData
  }

  def main(args: Array[String]): Unit = {

    kafaProducerParser.parse(args, TextKafkaProducerParams()).map { param =>

      // kafka config
      val props = new Properties()
      props.put("metadata.broker.list", param.brokerList)
      props.put("serializer.class", "kafka.serializer.StringEncoder")
//      props.put("serializer.class", "kafka.serializer.StringEncoder")
//      props.put("value.serializer",
//        "kafka.serializer.StringEncoder")
//            props.put("serializer.class",
//              classOf[ItemEncoder[Sample]].getName)
//            props.put("value.serializer",
//              classOf[ItemEncoder[Sample]].getName)
//      props.put("key.serializer",
//        classOf[StringEncoder].getName)
      props.put("producer.type", "async")

      // create producer
      val config = new ProducerConfig(props)
      //      val producer = new Producer[String, Sample](config)
      val producer = new Producer[String, String](config)

      // load messages
      val data = loadTestData(param.folder)
      // send
      var iter = data.iterator
      val batchsize = param.batchsize
      var count = 0
      var send_count = 0
      val batch: Array[Sample] = new Array[Sample](batchsize)
      //      data.map { sample => {
      ////        val data = new KeyedMessage[String, Sample](param.targetTopic, sample);
      //        val data = new KeyedMessage[String, String](param.targetTopic, sample.filename, sample.text);
      //        producer.send(data);
      //        send_count += 1
      //        println("Producer send text " + send_count)
      //        Thread.sleep(param.interval * 1000)
      //      }
      while (iter.hasNext) {
        try {
          if (count < batchsize) {
            batch(count) = iter.next()
            count += 1
          } else if (count == batchsize) {
//            batch.map { sample =>
//              println(sample.filename)
//            }
            // send
            producer.send(batch.map {sample =>
              new KeyedMessage[String, String](param.targetTopic, sample.filename, sample.text)
            }: _*)
            println("Producer send batch " + send_count)
            send_count += 1
            count = 0
            Thread.sleep(param.interval*1000)

          }

        } catch {
          case e: Exception => println(e)
        }
      }

    }
  }

}
